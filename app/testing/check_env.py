"""
Quick script to check if ChromaDB environment variables are set
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("\n" + "="*70)
print("ENVIRONMENT VARIABLES CHECK")
print("="*70 + "\n")

# Check ChromaDB variables
chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")

print(f"CHROMA_API_KEY:  {'✓ Set' if chroma_api_key else '✗ Not set'}")
if chroma_api_key:
    print(f"  Value: {chroma_api_key[:10]}..." if len(chroma_api_key) > 10 else f"  Value: {chroma_api_key}")

print(f"CHROMA_TENANT:   {'✓ Set' if chroma_tenant else '✗ Not set'}")
if chroma_tenant:
    print(f"  Value: {chroma_tenant}")

print(f"CHROMA_DATABASE: {'✓ Set' if chroma_database else '✗ Not set'}")
if chroma_database:
    print(f"  Value: {chroma_database}")

print("\n" + "="*70)

if chroma_api_key and chroma_tenant and chroma_database:
    print("✅ All ChromaDB Cloud variables are set!")
    print("   → Will use ChromaDB Cloud")
else:
    print("⚠️  ChromaDB Cloud variables missing!")
    print("   → Will use Local ChromaDB (vectorstore folder)")
    print("\nTo use Cloud ChromaDB:")
    print("1. Create a .env file in faq-chatbot/app/")
    print("2. Add these lines:")
    print("   CHROMA_API_KEY=your_api_key")
    print("   CHROMA_TENANT=your_tenant")
    print("   CHROMA_DATABASE=your_database")

print("="*70 + "\n")

