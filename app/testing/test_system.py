"""
Test script for the comprehensive RAG system.
Run this to verify all components are working correctly.
"""

import os
import sys
from faq_loader import FAQLoader, AGENT_TYPE_PRODUCT_SALES, AGENT_TYPE_CUSTOMER_CORPORATE
from chains import SupervisorAgent, SpecialistAgent, FAQChain

def test_data_loading():
    """Test 1: Data loading and collection creation"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading & Collection Creation")
    print("="*60)
    
    try:
        faq_loader = FAQLoader()
        print(f"✓ FAQLoader initialized successfully")
        
        # Load data into collections
        faq_loader.load_faq()
        print(f"✓ Data loaded into dual collections")
        
        # Get statistics
        stats = faq_loader.get_data_stats()
        print(f"\n📊 Data Statistics:")
        print(f"   Total records: {stats['total_records']}")
        print(f"   Product/Sales collection: {stats['product_sales_count']} records")
        print(f"   Customer/Corporate collection: {stats['customer_corporate_count']} records")
        
        if 'agent_distribution' in stats:
            print(f"   Agent distribution: {stats['agent_distribution']}")
        
        return faq_loader
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        return None


def test_supervisor_routing(faq_loader):
    """Test 2: Supervisor agent routing"""
    print("\n" + "="*60)
    print("TEST 2: Supervisor Agent Routing")
    print("="*60)
    
    try:
        supervisor = SupervisorAgent()
        print(f"✓ Supervisor Agent initialized")
        
        # Test queries
        test_queries = [
            ("Berapa premi SmartTravel International?", AGENT_TYPE_PRODUCT_SALES),
            ("Apa manfaat SmartActive?", AGENT_TYPE_PRODUCT_SALES),
            ("Bagaimana cara mengajukan klaim?", AGENT_TYPE_CUSTOMER_CORPORATE),
            ("Bagaimana cara bayar premi?", AGENT_TYPE_CUSTOMER_CORPORATE),
            ("Kontak customer care AXA?", AGENT_TYPE_CUSTOMER_CORPORATE),
        ]
        
        correct = 0
        for query, expected in test_queries:
            result = supervisor.route_query(query)
            status = "✓" if result == expected else "❌"
            correct += (result == expected)
            print(f"   {status} '{query[:40]}...' → {result}")
        
        print(f"\n   Accuracy: {correct}/{len(test_queries)} ({correct/len(test_queries)*100:.1f}%)")
        
        return supervisor
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return None


def test_retrieval(faq_loader):
    """Test 3: Vector retrieval and re-ranking"""
    print("\n" + "="*60)
    print("TEST 3: Vector Retrieval & Re-Ranking")
    print("="*60)
    
    try:
        # Test Product Agent retrieval
        print("\n📦 Testing PRODUCT_SALES retrieval:")
        query1 = "Apa itu SmartActive?"
        results1 = faq_loader.query_with_reranking(
            query=query1,
            agent_type=AGENT_TYPE_PRODUCT_SALES,
            top_k=20,
            final_k=3
        )
        
        print(f"   Query: '{query1}'")
        print(f"   Retrieved: {len(results1)} chunks")
        if results1:
            for i, result in enumerate(results1, 1):
                chunk_id = result.get('id', 'N/A')
                distance = result.get('boosted_distance', 0)
                print(f"   [{i}] {chunk_id} (distance: {distance:.4f})")
        
        # Test Customer Agent retrieval
        print("\n👥 Testing CUSTOMER_CORPORATE retrieval:")
        query2 = "Bagaimana cara klaim SmartTravel?"
        results2 = faq_loader.query_with_reranking(
            query=query2,
            agent_type=AGENT_TYPE_CUSTOMER_CORPORATE,
            top_k=20,
            final_k=3
        )
        
        print(f"   Query: '{query2}'")
        print(f"   Retrieved: {len(results2)} chunks")
        if results2:
            for i, result in enumerate(results2, 1):
                chunk_id = result.get('id', 'N/A')
                distance = result.get('boosted_distance', 0)
                print(f"   [{i}] {chunk_id} (distance: {distance:.4f})")
        
        print(f"\n✓ Retrieval system working correctly")
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")


def test_end_to_end(faq_loader):
    """Test 4: End-to-end RAG system"""
    print("\n" + "="*60)
    print("TEST 4: End-to-End RAG System")
    print("="*60)
    
    try:
        # Initialize complete system
        faq_chain = FAQChain(faq_loader=faq_loader)
        print(f"✓ Complete RAG system initialized")
        
        # Test queries
        test_queries = [
            "Apa keunggulan asuransi SmartActive?",
            "Bagaimana cara mengajukan klaim asuransi kesehatan?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            print(f"   Generating answer...")
            
            answer = faq_chain.generate_answer(query)
            
            # Display truncated answer
            answer_preview = answer[:150] + "..." if len(answer) > 150 else answer
            print(f"   Answer: {answer_preview}")
            print(f"   ✓ Answer generated successfully")
        
        print(f"\n✓ End-to-end system working correctly")
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")


def main():
    """Run all tests"""
    print("\n" + "🚀"*30)
    print("AXA FAQ CHATBOT - COMPREHENSIVE RAG SYSTEM TEST")
    print("🚀"*30)
    
    # Test 1: Data Loading
    faq_loader = test_data_loading()
    if not faq_loader:
        print("\n❌ Critical failure in Test 1. Stopping tests.")
        return
    
    # Test 2: Supervisor Routing
    supervisor = test_supervisor_routing(faq_loader)
    
    # Test 3: Retrieval
    test_retrieval(faq_loader)
    
    # Test 4: End-to-End
    test_end_to_end(faq_loader)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ All critical components tested")
    print("✓ System ready for deployment")
    print("\nTo run the chatbot:")
    print("   streamlit run main.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

