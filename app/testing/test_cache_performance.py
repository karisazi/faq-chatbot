#!/usr/bin/env python3
"""
Quick test to demonstrate query caching performance improvements.
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faq_loader import FAQLoader
from chains import FAQChain

def test_cache_performance():
    """Test query performance with and without cache."""
    print("="*70)
    print("🧪 TESTING QUERY CACHE PERFORMANCE")
    print("="*70)
    
    # Initialize system
    print("\n📚 Step 1: Initializing FAQ system...")
    faq = FAQLoader()
    faq.load_faq()
    llm = FAQChain(faq_loader=faq)
    
    # Test query
    test_query = "Apa itu SmartHome?"
    
    # First query (no cache)
    print(f"\n🔍 Step 2: First query (no cache)")
    print(f"   Query: '{test_query}'")
    start = time.time()
    answer1 = llm.generate_answer(test_query)
    time1 = time.time() - start
    print(f"   ⏱️  Time: {time1:.2f}s")
    print(f"   Answer length: {len(answer1)} chars")
    
    # Second query (cached)
    print(f"\n⚡ Step 3: Second query (should be cached)")
    print(f"   Query: '{test_query}'")
    start = time.time()
    answer2 = llm.generate_answer(test_query)
    time2 = time.time() - start
    print(f"   ⏱️  Time: {time2:.2f}s")
    print(f"   Answer length: {len(answer2)} chars")
    
    # Results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"First query (no cache):  {time1:.2f}s")
    print(f"Second query (cached):   {time2:.2f}s")
    
    if time2 < 0.1:
        speedup = time1 / time2
        print(f"\n✅ CACHE WORKING! Speedup: {speedup:.0f}x faster")
        print(f"   Second query was nearly instant! ({time2:.3f}s)")
    elif time2 < time1 * 0.5:
        speedup = time1 / time2
        print(f"\n✅ Some improvement: {speedup:.1f}x faster")
    else:
        print(f"\n⚠️  Cache may not be working properly")
    
    # Test warm-up feature
    print("\n" + "="*70)
    print("🔥 TESTING CACHE WARM-UP")
    print("="*70)
    
    common_queries = [
        "Berapa premi SmartTravel?",
        "Bagaimana cara klaim?",
        "Nomor customer care AXA"
    ]
    
    print(f"\n🔥 Warming up cache with {len(common_queries)} queries...")
    warm_start = time.time()
    llm.warm_up_cache(common_queries)
    warm_time = time.time() - warm_start
    print(f"   ⏱️  Warm-up took: {warm_time:.2f}s")
    
    # Test cached queries
    print(f"\n⚡ Testing warmed-up queries...")
    for query in common_queries:
        start = time.time()
        llm.generate_answer(query)
        query_time = time.time() - start
        status = "✅" if query_time < 0.1 else "⚠️"
        print(f"   {status} '{query[:40]}...' - {query_time:.3f}s")
    
    print("\n" + "="*70)
    print("✅ CACHE PERFORMANCE TEST COMPLETE")
    print("="*70)
    print(f"\nKey Benefits:")
    print(f"  • Repeated queries: ~instant (<0.1s)")
    print(f"  • Cache warm-up: Pre-loads common FAQs")
    print(f"  • Memory efficient: Max 100 cached queries")

if __name__ == "__main__":
    try:
        test_cache_performance()
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

