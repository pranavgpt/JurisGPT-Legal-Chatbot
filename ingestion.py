import os
import re
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

def extract_ipc_sections(text):
    """Extract IPC section numbers from text"""
    # Pattern to match IPC sections like "Section 302", "Sec 420", "IPC 376", etc.
    patterns = [
        r'Section\s+(\d+[A-Z]*)',
        r'Sec\.?\s+(\d+[A-Z]*)',
        r'IPC\s+(\d+[A-Z]*)',
        r'§\s*(\d+[A-Z]*)'
    ]
    
    sections = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sections.update(matches)
    
    return list(sections)

def categorize_legal_content(text):
    """Categorize legal content type"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['murder', 'homicide', 'death', 'killing']):
        return 'homicide'
    elif any(word in text_lower for word in ['theft', 'robbery', 'burglary', 'stealing']):
        return 'property_crime'
    elif any(word in text_lower for word in ['assault', 'battery', 'violence', 'hurt']):
        return 'assault'
    elif any(word in text_lower for word in ['fraud', 'cheating', 'forgery', 'counterfeiting']):
        return 'fraud'
    elif any(word in text_lower for word in ['rape', 'sexual', 'molestation', 'harassment']):
        return 'sexual_offense'
    elif any(word in text_lower for word in ['corruption', 'bribery', 'public servant']):
        return 'corruption'
    else:
        return 'general'

def enhance_document_metadata(docs):
    """Enhance documents with better metadata for legal analysis"""
    enhanced_docs = []
    
    for doc in docs:
        # Extract IPC sections
        ipc_sections = extract_ipc_sections(doc.page_content)
        
        # Categorize content
        category = categorize_legal_content(doc.page_content)
        
        # Enhanced metadata
        enhanced_metadata = {
            **doc.metadata,
            'ipc_sections': ipc_sections,
            'category': category,
            'content_type': 'legal_document',
            'word_count': len(doc.page_content.split())
        }
        
        # Create enhanced document
        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata=enhanced_metadata
        )
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs

def create_ipc_reference_documents():
    """Create structured IPC reference documents"""
    ipc_sections = {
        "302": {
            "title": "Murder",
            "description": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            "elements": ["Intention to cause death", "Knowledge that act is likely to cause death", "Actual death occurs"],
            "punishment": "Death or life imprisonment + fine",
            "bailable": False,
            "cognizable": True
        },
        "376": {
            "title": "Rape",
            "description": "Whoever commits rape shall be punished with imprisonment of either description for a term which shall not be less than seven years.",
            "elements": ["Sexual intercourse", "Against will/without consent", "With woman"],
            "punishment": "Minimum 7 years imprisonment",
            "bailable": False,
            "cognizable": True
        },
        "420": {
            "title": "Cheating and dishonestly inducing delivery of property",
            "description": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property.",
            "elements": ["Deception", "Dishonest inducement", "Delivery of property"],
            "punishment": "Imprisonment up to 7 years + fine",
            "bailable": False,
            "cognizable": True
        },
        "379": {
            "title": "Theft",
            "description": "Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
            "elements": ["Dishonest intention", "Movable property", "Taking out of possession"],
            "punishment": "Up to 3 years imprisonment or fine or both",
            "bailable": True,
            "cognizable": True
        },
        "323": {
            "title": "Punishment for voluntarily causing hurt",
            "description": "Whoever voluntarily causes hurt shall be punished with imprisonment of either description for a term which may extend to one year, or with fine which may extend to one thousand rupees, or with both.",
            "elements": ["Voluntary action", "Causing hurt", "No grievous hurt"],
            "punishment": "Up to 1 year imprisonment or fine up to Rs. 1000 or both",
            "bailable": True,
            "cognizable": True
        }
        # Add more sections as needed
    }
    
    reference_docs = []
    for section_num, details in ipc_sections.items():
        content = f"""
IPC Section {section_num}: {details['title']}

Description: {details['description']}

Essential Elements:
{chr(10).join(f"- {element}" for element in details['elements'])}

Punishment: {details['punishment']}
Bailable: {'Yes' if details['bailable'] else 'No'}
Cognizable: {'Yes' if details['cognizable'] else 'No'}

This section is commonly applied in cases involving {details['title'].lower()} and related offenses.
        """
        
        doc = Document(
            page_content=content.strip(),
            metadata={
                'source': f'IPC_Section_{section_num}',
                'ipc_sections': [section_num],
                'category': 'ipc_reference',
                'content_type': 'legal_statute',
                'section_title': details['title'],
                'bailable': details['bailable'],
                'cognizable': details['cognizable']
            }
        )
        reference_docs.append(doc)
    
    return reference_docs

def embed_and_save_documents():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load PDF documents
    loader = PyPDFDirectoryLoader("./LEGAL-DATA")
    print("Loading PDF documents...")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Slightly larger chunks for legal context
        chunk_overlap=300,  # More overlap to preserve legal context
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    split_documents = text_splitter.split_documents(docs)
    print(f"Split into {len(split_documents)} chunks")
    
    # Enhance metadata
    enhanced_documents = enhance_document_metadata(split_documents)
    print("Enhanced document metadata")
    
    # Create IPC reference documents
    ipc_reference_docs = create_ipc_reference_documents()
    print(f"Created {len(ipc_reference_docs)} IPC reference documents")
    
    # Combine all documents
    all_documents = enhanced_documents + ipc_reference_docs
    print(f"Total documents: {len(all_documents)}")
    
    # Create vector store in batches
    batch_size = 100
    batched_documents = [all_documents[i:i + batch_size] for i in range(0, len(all_documents), batch_size)]
    
    print(f"Creating {len(batched_documents)} batches...")
    vector_stores = []
    
    for i, batch in enumerate(batched_documents):
        print(f"Processing batch {i+1}/{len(batched_documents)}")
        vector_store = FAISS.from_documents(batch, embeddings)
        vector_stores.append(vector_store)
    
    # Merge vector stores
    print("Merging vector stores...")
    vectors = vector_stores[0]
    for vector_store in vector_stores[1:]:
        vectors.merge_from(vector_store)
    
    # Save the vector store
    vectors.save_local("my_vector_store")
    print("Enhanced vector store saved successfully!")
    
    # Print statistics
    categories = {}
    ipc_sections = set()
    for doc in all_documents:
        cat = doc.metadata.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        ipc_sections.update(doc.metadata.get('ipc_sections', []))
    
    print("\nDocument Statistics:")
    print(f"Categories: {categories}")
    print(f"IPC Sections found: {len(ipc_sections)}")
    print(f"Sample IPC sections: {list(ipc_sections)[:10]}")

if __name__ == "__main__":
    embed_and_save_documents()