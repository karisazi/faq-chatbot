#!/usr/bin/env python3
"""
Test script to demonstrate enhanced boosting mechanism.
Tests the specific case: "coba jelaskan mengenai produk SmartHome"
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faq_loader import FAQLoader, AGENT_TYPE_PRODUCT_SALES

# Configure logging to see detailed boost information
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

def test_smarthome_query():
    """
    Test Case: Query for SmartHome product should return SmartHome docs in top results
    """
    print("="*70)
    print("🧪 TESTING ENHANCED BOOSTING MECHANISM")
    print("="*70)
    
    # Initialize FAQ Loader
    print("\n📚 Step 1: Initializing FAQLoader...")
    faq_loader = FAQLoader(filenames=["AXA_QNA.csv"])
    
    # Load FAQ data
    print("📥 Step 2: Loading FAQ data into vector database...")
    faq_loader.load_faq()
    
    # Test Query
    test_query = "coba jelaskan mengenai produk SmartHome"
    print(f"\n🔍 Step 3: Testing query: '{test_query}'")
    print("-"*70)
    
    # Query with enhanced re-ranking
    print("\n🚀 Executing query with ENHANCED re-ranking (top_k=20)...")
    results = faq_loader.query_with_reranking(
        query=test_query,
        agent_type=AGENT_TYPE_PRODUCT_SALES,
        top_k=20,  # Fetch 20 candidates
        final_k=5   # Return top 5 after re-ranking
    )
    
    # Display Results
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    if not results:
        print("❌ No results found!")
        return False
    
    print(f"\n✅ Retrieved {len(results)} results after re-ranking:")
    
    smarthome_count = 0
    for idx, result in enumerate(results, 1):
        chunk_id = result.get('id', 'N/A')
        metadata = result.get('metadata', {})
        product_name = metadata.get('product_name', '')
        distance = result.get('distance', 0)
        boosted_distance = result.get('boosted_distance', 0)
        boost_score = result.get('boost_score', 0)
        boost_reasons = result.get('boost_reasons', [])
        question = result.get('question', '')
        
        # Count SmartHome results
        if 'smarthome' in product_name.lower():
            smarthome_count += 1
            emoji = "✅"
        else:
            emoji = "⚠️"
        
        print(f"\n{emoji} Result #{idx}")
        print(f"   ID: {chunk_id}")
        print(f"   Product: {product_name}")
        print(f"   Original Distance: {distance:.4f}")
        print(f"   Boosted Distance: {boosted_distance:.4f}")
        print(f"   Boost Score: {boost_score:.4f}")
        print(f"   Boost Reasons: {', '.join(boost_reasons) if boost_reasons else 'None'}")
        print(f"   Has Question: {'Yes' if question else 'No'}")
        
        # Show snippet of text
        text = metadata.get('text_original', '')
        if text:
            snippet = text[:100] + "..." if len(text) > 100 else text
            print(f"   Text: {snippet}")
    
    # Success Criteria
    print("\n" + "="*70)
    print("🎯 SUCCESS CRITERIA")
    print("="*70)
    
    success = True
    
    print(f"\n1. SmartHome products in top 5: {smarthome_count}/5")
    if smarthome_count >= 3:
        print("   ✅ PASS - At least 3 SmartHome results in top 5")
    else:
        print("   ❌ FAIL - Expected at least 3 SmartHome results")
        success = False
    
    # Check if top result is SmartHome
    top_result = results[0]
    top_product = top_result.get('metadata', {}).get('product_name', '').lower()
    print(f"\n2. Top result is SmartHome: {'smarthome' in top_product}")
    if 'smarthome' in top_product:
        print("   ✅ PASS - Top result is SmartHome product")
    else:
        print(f"   ⚠️  WARNING - Top result is '{top_product}' (might still be relevant)")
    
    # Check if boosting is applied
    boost_applied = any(r.get('boost_score', 0) < -0.1 for r in results)
    print(f"\n3. Boosting applied: {boost_applied}")
    if boost_applied:
        print("   ✅ PASS - Boost scores are being applied")
    else:
        print("   ❌ FAIL - No significant boost scores detected")
        success = False
    
    print("\n" + "="*70)
    if success:
        print("✅ TEST PASSED - Enhanced boosting is working correctly!")
    else:
        print("❌ TEST FAILED - Boosting mechanism needs adjustment")
    print("="*70)
    
    return success


def test_additional_queries():
    """Test additional product queries"""
    print("\n\n" + "="*70)
    print("🧪 ADDITIONAL TEST QUERIES")
    print("="*70)
    
    faq_loader = FAQLoader(filenames=["AXA_QNA.csv"])
    faq_loader.load_faq()
    
    test_queries = [
        ("SmartActive itu apa?", "smartactive"),
        ("jelaskan asuransi rumah tinggal", "smarthome"),
        ("produk untuk perjalanan", "smarttravel"),
    ]
    
    for query, expected_product in test_queries:
        print(f"\n📝 Query: '{query}'")
        print(f"   Expected: {expected_product}")
        
        results = faq_loader.query_with_reranking(
            query=query,
            agent_type=AGENT_TYPE_PRODUCT_SALES,
            top_k=20,
            final_k=3
        )
        
        if results:
            top_product = results[0].get('metadata', {}).get('product_name', '').lower()
            if expected_product in top_product:
                print(f"   ✅ PASS - Top result: {top_product}")
            else:
                print(f"   ⚠️  WARNING - Top result: {top_product} (expected: {expected_product})")
        else:
            print(f"   ❌ FAIL - No results found")


if __name__ == "__main__":
    try:
        # Main test
        success = test_smarthome_query()
        
        # Additional tests
        # test_additional_queries()  # Uncomment to run additional tests
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

