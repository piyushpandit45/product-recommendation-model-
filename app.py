import os
from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from collections import defaultdict
import re

app = Flask(__name__, template_folder='templates', static_folder='static')

# Helper function to clean data
def clean_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

# Modified to read from CSV with image URLs
def extract_data_from_csv():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
        df = pd.read_csv(csv_path)
        
        # Clean and prepare the data
        df = df.rename(columns={
            'title': 'Product_Name',
            'genre': 'Category',
            'overview': 'Description',
            'popularity': 'Price',
            'user_id': 'User_ID',
            'image_url': 'Image_URL'
        })
        
        # Add missing columns and clean data
        df['Product_ID'] = 'P' + df.index.astype(str).str.zfill(3)
        df['Brand'] = df['Category'].str.split(':').str[-1].str.strip()
        df['Category'] = df['Category'].str.split(':').str[0].str.strip()
        df['Transaction'] = 'T' + df.index.astype(str).str.zfill(3)
        
        # Clean all text fields
        text_columns = ['Product_Name', 'Category', 'Brand', 'Description']
        for col in text_columns:
            df[col] = df[col].apply(clean_text)
        
        # Ensure Image_URL has valid values
        df['Image_URL'] = df['Image_URL'].fillna('https://via.placeholder.com/150')
        
        return df
        
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        return pd.DataFrame()

# Recommendation system
class RecommenderSystem:
    def __init__(self, df):
        self.df = df
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_models(self):
        # User similarity
        user_product_matrix = self.df.pivot_table(
            index='User_ID',
            columns='Product_ID',
            values='Transaction',
            aggfunc='count',
            fill_value=0
        )
        
        user_similarity = cosine_similarity(user_product_matrix)
        self.user_sim = pd.DataFrame(
            user_similarity,
            index=user_product_matrix.index,
            columns=user_product_matrix.index
        )
        
        # Product similarity
        self.df['product_features'] = (
            self.df['Category'] + ' ' + 
            self.df['Brand'] + ' ' + 
            self.df['Product_Name'] + ' ' + 
            self.df['Description']
        )
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['product_features'].dropna())
        
        product_similarity = cosine_similarity(tfidf_matrix)
        product_ids = self.df['Product_ID'].unique()
        self.product_sim = pd.DataFrame(
            product_similarity,
            index=product_ids,
            columns=product_ids
        )
        
        # Save models
        with open(os.path.join(self.models_dir, 'user_sim.pkl'), 'wb') as f:
            pickle.dump(self.user_sim, f)
        with open(os.path.join(self.models_dir, 'product_sim.pkl'), 'wb') as f:
            pickle.dump(self.product_sim, f)
    
    def load_models(self):
        try:
            with open(os.path.join(self.models_dir, 'user_sim.pkl'), 'rb') as f:
                self.user_sim = pickle.load(f)
            with open(os.path.join(self.models_dir, 'product_sim.pkl'), 'rb') as f:
                self.product_sim = pickle.load(f)
            return True
        except:
            return False
    
    def recommend(self, user_id, top_n=5):
        try:
            # Get similar users
            if user_id not in self.user_sim.index:
                return pd.DataFrame()
                
            similar_users = self.user_sim[user_id].sort_values(ascending=False)[1:6].index
            
            # Get products from similar users
            similar_users_products = self.df[
                self.df['User_ID'].isin(similar_users)
            ]['Product_ID'].unique()
            
            # Get user's own products
            user_products = self.df[
                self.df['User_ID'] == user_id
            ]['Product_ID'].unique()
            
            # Recommend new products
            recommendations = list(set(similar_users_products) - set(user_products))
            
            # If not enough, use product similarity
            if len(recommendations) < top_n:
                product_scores = defaultdict(float)
                
                for product in user_products:
                    if product in self.product_sim.columns:
                        similar_products = self.product_sim[product].sort_values(ascending=False)[1:top_n+1]
                        for p, score in similar_products.items():
                            if p not in user_products:
                                product_scores[p] += score
                
                sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
                additional_recs = [p[0] for p in sorted_products[:top_n - len(recommendations)]]
                recommendations.extend(additional_recs)
            
            # Get product details with images
            result = self.df[
                (self.df['Product_ID'].isin(recommendations[:top_n])) &
                (~self.df.duplicated('Product_ID'))
            ]
            
            return result[['Product_ID', 'Product_Name', 'Category', 'Brand', 'Price', 'Description', 'Image_URL']]
            
        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return pd.DataFrame()

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def home():
    df = extract_data_from_csv()
    
    if df.empty:
        return render_template('index.html', 
                            error="Failed to load product data",
                            user_ids=[])
    
    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip().upper()
        
        if not user_id:
            return render_template('index.html', 
                                error="Please enter a user ID",
                                user_ids=df['User_ID'].unique()[:10])
        
        if user_id not in df['User_ID'].unique():
            return render_template('index.html',
                                error=f"User {user_id} not found. Try: {', '.join(df['User_ID'].unique()[:5])}...",
                                user_ids=df['User_ID'].unique()[:10])
        
        recommender = RecommenderSystem(df)
        
        if not recommender.load_models():
            recommender.prepare_models()
        
        recommendations = recommender.recommend(user_id)
        
        if recommendations.empty:
            return render_template('index.html',
                                error="No recommendations found for this user",
                                user_ids=df['User_ID'].unique()[:10])
        
        return render_template('results.html',
                            user_id=user_id,
                            recommendations=recommendations.to_dict('records'))
    
    return render_template('index.html', user_ids=df['User_ID'].unique()[:10])

if __name__ == '__main__':
    app.run(debug=True)