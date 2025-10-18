"""
SBERT + FAISS Similarity Engine

This module implements semantic similarity matching using:
- Sentence-BERT for generating embeddings
- FAISS for efficient similarity search
- Cosine similarity for matching resumes to job descriptions

Key Features:
- Fast vector similarity search
- Scalable to large datasets
- Semantic understanding beyond keyword matching
"""

import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """
    Handles semantic similarity computation using SBERT and FAISS
    """
    
    def __init__(self, model_name='all-mpnet-base-v2', index_path='faiss_indexes/resume_index'):
        """
        Initialize the similarity engine
        
        Args:
            model_name (str): Sentence-BERT model name
            index_path (str): Path to save/load FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.model = None
        self.faiss_index = None
        self.document_embeddings = []
        self.document_metadata = []
        
        # Initialize the sentence transformer model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading SentenceBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("SentenceBERT model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading SentenceBERT model: {str(e)}")
            raise
    
    def encode_texts(self, texts, batch_size=32):
        """
        Encode texts into embeddings using SBERT
        
        Args:
            texts (list): List of text documents
            batch_size (int): Batch size for encoding
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            logger.info(f"Encoding {len(texts)} texts with SBERT...")
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def build_faiss_index(self, texts, metadata=None):
        """
        Build FAISS index from text documents
        
        Args:
            texts (list): List of text documents
            metadata (list): Optional metadata for each document
        """
        try:
            logger.info("Building FAISS index...")
            
            # Generate embeddings
            embeddings = self.encode_texts(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            
            # Add embeddings to index
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store embeddings and metadata
            self.document_embeddings = embeddings
            self.document_metadata = metadata or list(range(len(texts)))
            
            logger.info(f"FAISS index built with {self.faiss_index.ntotal} documents")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise
    
    def save_index(self, index_path=None):
        """Save FAISS index and metadata to disk"""
        try:
            save_path = index_path or self.index_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, f"{save_path}.faiss")
            
            # Save metadata and embeddings
            with open(f"{save_path}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'metadata': self.document_metadata,
                    'embeddings': self.document_embeddings
                }, f)
            
            logger.info(f"FAISS index saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def load_index(self, index_path=None):
        """Load FAISS index and metadata from disk"""
        try:
            load_path = index_path or self.index_path
            
            # Load FAISS index
            if os.path.exists(f"{load_path}.faiss"):
                self.faiss_index = faiss.read_index(f"{load_path}.faiss")
                
                # Load metadata and embeddings
                with open(f"{load_path}_metadata.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.document_metadata = data['metadata']
                    self.document_embeddings = data['embeddings']
                
                logger.info(f"FAISS index loaded from {load_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False
    
    def search_similar(self, query_text, k=5, threshold=0.5):
        """
        Search for similar documents using FAISS
        
        Args:
            query_text (str): Query text
            k (int): Number of similar documents to return
            threshold (float): Minimum similarity threshold
            
        Returns:
            list: List of tuples (similarity_score, document_metadata)
        """
        try:
            if self.faiss_index is None:
                raise ValueError("FAISS index not built or loaded")
            
            # Encode query
            query_embedding = self.encode_texts([query_text])
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), k
            )
            
            # Filter results by threshold and format
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= threshold:
                    results.append({
                        'similarity_score': float(similarity),
                        'document_id': int(idx),
                        'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else None,
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(results)} similar documents above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            raise
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1 (str): First text (e.g., resume)
            text2 (str): Second text (e.g., job description)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Encode both texts
            embeddings = self.encode_texts([text1, text2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(
                embeddings[0].reshape(1, -1), 
                embeddings[1].reshape(1, -1)
            )
            
            similarity_score = float(similarity_matrix[0][0])
            
            logger.info(f"Calculated similarity score: {similarity_score:.3f}")
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise
    
    def batch_similarity(self, resume_texts, job_descriptions):
        """
        Calculate similarity scores for multiple resume-job pairs
        
        Args:
            resume_texts (list): List of resume texts
            job_descriptions (list): List of job descriptions
            
        Returns:
            numpy.ndarray: Similarity matrix (resumes x jobs)
        """
        try:
            logger.info(f"Calculating batch similarities for {len(resume_texts)} resumes and {len(job_descriptions)} jobs")
            
            # Encode all texts
            resume_embeddings = self.encode_texts(resume_texts)
            job_embeddings = self.encode_texts(job_descriptions)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
            
            logger.info(f"Generated similarity matrix shape: {similarity_matrix.shape}")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {str(e)}")
            raise
    
    def get_match_explanation(self, text1, text2, similarity_score):
        """
        Generate explanation for match quality
        
        Args:
            text1 (str): Resume text
            text2 (str): Job description text  
            similarity_score (float): Calculated similarity score
            
        Returns:
            dict: Match explanation with quality level and reasoning
        """
        try:
            # Determine match quality
            if similarity_score >= 0.75:
                quality = "Excellent"
                explanation = "Very high semantic similarity. Strong alignment between skills and requirements."
            elif similarity_score >= 0.60:
                quality = "Good" 
                explanation = "Good semantic match. Most requirements align well with candidate profile."
            elif similarity_score >= 0.40:
                quality = "Fair"
                explanation = "Moderate match. Some requirements met but significant gaps exist."
            else:
                quality = "Poor"
                explanation = "Low semantic similarity. Major misalignment between profile and requirements."
            
            # Extract key matching terms (simplified approach)
            resume_words = set(text1.lower().split())
            job_words = set(text2.lower().split())
            common_words = resume_words.intersection(job_words)
            
            # Filter common technical terms
            technical_terms = [word for word in common_words 
                             if len(word) > 3 and word.isalpha()]
            
            return {
                'quality': quality,
                'score': round(similarity_score * 100, 1),
                'explanation': explanation,
                'common_terms': list(technical_terms)[:10],  # Top 10 common terms
                'recommendations': self._get_recommendations(quality)
            }
            
        except Exception as e:
            logger.error(f"Error generating match explanation: {str(e)}")
            return {
                'quality': 'Unknown',
                'score': round(similarity_score * 100, 1),
                'explanation': 'Unable to generate detailed explanation',
                'common_terms': [],
                'recommendations': []
            }
    
    def _get_recommendations(self, quality):
        """Get improvement recommendations based on match quality"""
        recommendations = {
            'Excellent': [
                "Your profile is an excellent match! Consider highlighting specific achievements.",
                "Tailor your experience descriptions to match the job's language."
            ],
            'Good': [
                "Good match overall. Consider adding specific skills mentioned in the job posting.",
                "Highlight relevant projects that demonstrate required competencies."
            ],
            'Fair': [
                "Consider developing skills in key areas mentioned in the job description.",
                "Add relevant certifications or training to strengthen your profile."
            ],
            'Poor': [
                "Significant skill gaps identified. Consider upskilling in required areas.",
                "Focus on building experience in technologies mentioned in the job posting."
            ]
        }
        
        return recommendations.get(quality, ["Review job requirements and align your profile accordingly."])
    
    def update_index_with_new_documents(self, new_texts, new_metadata=None):
        """
        Add new documents to existing FAISS index
        
        Args:
            new_texts (list): List of new text documents
            new_metadata (list): Metadata for new documents
        """
        try:
            if self.faiss_index is None:
                raise ValueError("No existing index found. Build index first.")
            
            # Encode new texts
            new_embeddings = self.encode_texts(new_texts)
            
            # Add to FAISS index
            self.faiss_index.add(new_embeddings.astype('float32'))
            
            # Update stored data
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
            new_meta = new_metadata or list(range(len(new_texts)))
            self.document_metadata.extend(new_meta)
            
            logger.info(f"Added {len(new_texts)} new documents to index")
            
        except Exception as e:
            logger.error(f"Error updating index: {str(e)}")
            raise