import streamlit as st
import PyPDF2
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import easyocr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

class PDFRAGSystem:
    def __init__(self):
        # Initialize OCR reader
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Initialize embedding model
        self.embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Hugging Face Pipeline for LLM
        hf_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", max_length=1024)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def extract_text_from_image(self, image_file):
        """Extract text from uploaded image using EasyOCR"""
        # Convert uploaded file to image array
        image = Image.open(image_file)
        image = np.array(image)
        
        # Perform OCR
        with st.spinner('Performing OCR on image...'):
            results = self.reader.readtext(image)
        
        # Combine all detected text
        text = ' '.join([result[1] for result in results])
        
        # Show detected text in UI for verification
        with st.expander("View detected text"):
            st.write(text)
            
        return text

    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def generate_embeddings(self, chunks):
        """Generate embeddings for text chunks"""
        embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
                embedding = self.embed_model(**inputs).last_hidden_state.mean(dim=1)
                embeddings.append(embedding.numpy().flatten())
        
        return np.array(embeddings)
    
    def create_faiss_index(self, embeddings):
        """Create FAISS index for efficient similarity search"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def retrieve_relevant_context(self, query, index, chunks, top_k=3):
        """Retrieve most relevant text chunks"""
        # Generate query embedding
        with torch.no_grad():
            query_inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
            query_embedding = self.embed_model(**query_inputs).last_hidden_state.mean(dim=1).numpy()
        
        # Search similar chunks
        D, I = index.search(query_embedding, top_k)
        relevant_chunks = [chunks[i] for i in I[0]]
        return relevant_chunks
    
    def generate_answer(self, query, context):
        """Generate answer using retrieved context"""
        prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm(prompt)
        return response
    
    def run_rag_pipeline(self, uploaded_file, query):
        """Main RAG pipeline"""
        # Extract text based on file type
        file_type = uploaded_file.type
        st.info(f"Processing {file_type} file...")
        
        try:
            if file_type == 'application/pdf':
                text = self.extract_text_from_pdf(uploaded_file)
            elif file_type in ['image/jpeg', 'image/png', 'image/jpg']:
                text = self.extract_text_from_image(uploaded_file)
            else:
                raise ValueError("Unsupported file type. Please upload a PDF or image.")

            if not text.strip():
                raise ValueError("No text was extracted from the file. Please check the file content.")

            # Show progress
            progress_bar = st.progress(0)
            
            # Chunk text
            st.info("Chunking text...")
            progress_bar.progress(20)
            chunks = self.chunk_text(text)
            
            # Generate embeddings
            st.info("Generating embeddings...")
            progress_bar.progress(40)
            embeddings = self.generate_embeddings(chunks)
            
            # Create FAISS index
            st.info("Creating search index...")
            progress_bar.progress(60)
            index = self.create_faiss_index(embeddings)
            
            # Retrieve context
            st.info("Retrieving relevant context...")
            progress_bar.progress(80)
            context = self.retrieve_relevant_context(query, index, chunks)
            
            # Generate answer
            st.info("Generating answer...")
            progress_bar.progress(100)
            answer = self.generate_answer(query, context)
            
            return answer
            
        except Exception as e:
            st.error(f"Error in RAG pipeline: {str(e)}")
            raise e

def main():
    st.title("NutriScan AI")
    st.write("Upload a food label image or PDF and ask questions about its nutritional content.")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload PDF or Image", type=['pdf', 'jpeg', 'jpg', 'png'])
    
    # Query Input
    query = st.text_input("Enter your question")
    
    # Default Query Button
    use_default_query = st.button("Use Default Query")
    default_query = """Provide a comprehensive analysis of the food label, including:
    1. Nutritional Content Overview
    2. Key Ingredients Breakdown
    3. Potential Health Benefits
    4. Potential Health Concerns
    5. Recommended Daily Intake
    6. Nutritional Recommendations"""
    
    if 'rag_system' not in st.session_state:
        with st.spinner('Initializing NutriScan AI...'):
            st.session_state.rag_system = PDFRAGSystem()
    
    if uploaded_file:
        try:
            # Determine the query to use
            if use_default_query:
                st.write(f"Using default query: {default_query}")
                query = default_query
            
            if query:
                with st.spinner('Processing your request...'):
                    answer = st.session_state.rag_system.run_rag_pipeline(uploaded_file, query)
                st.success("Analysis complete!")
                st.write("Answer:", answer)
            else:
                st.warning("Please enter a question or click 'Use Default Query'.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("If this error persists, please try with a different image or ensure the image contains clear, readable text.")

if __name__ == "__main__":
    main()
