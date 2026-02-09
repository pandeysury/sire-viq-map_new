#!/usr/bin/env python3
"""
Test script for VIQ AI Matching System
"""
import requests
import json
import time
import sys
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the backend is running.")
        return False

def test_vessel_types():
    """Test vessel types endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/vessel-types")
        if response.status_code == 200:
            data = response.json()
            vessel_types = data.get("vessel_types", [])
            print(f"✅ Vessel types loaded: {vessel_types}")
            return True
        else:
            print(f"❌ Vessel types test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Vessel types test error: {str(e)}")
        return False

def test_stats():
    """Test stats endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats: {data}")
            return True
        else:
            print(f"❌ Stats test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats test error: {str(e)}")
        return False

def test_search():
    """Test search functionality"""
    test_queries = [
        {
            "query": "Inert gas system not properly maintained according to ISGOTT guidelines",
            "vessel_type": "Oil",
            "top_k": 3
        },
        {
            "query": "Fire detection system malfunction in engine room",
            "vessel_type": "All",
            "top_k": 3
        },
        {
            "query": "Crew training records incomplete for cargo operations",
            "vessel_type": "Chemical",
            "top_k": 2
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n🔍 Test Query {i}: {query['query'][:50]}...")
            response = requests.post(f"{API_BASE_URL}/search", json=query)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get("matches", [])
                print(f"✅ Found {len(matches)} matches")
                
                for j, match in enumerate(matches[:2], 1):
                    print(f"   {j}. VIQ {match['viq_number']}: {match['similarity_score']:.1%} match")
                    print(f"      {match['question'][:80]}...")
            else:
                print(f"❌ Search test {i} failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Search test {i} error: {str(e)}")
            return False
    
    return True

def test_ai_analysis():
    """Test AI analysis functionality"""
    query = {
        "query": "Safety equipment inspection overdue and certificates expired",
        "vessel_type": "LPG",
        "top_k": 3
    }
    
    try:
        print(f"\n🧠 AI Analysis Test: {query['query']}")
        response = requests.post(f"{API_BASE_URL}/analyze-finding", json=query)
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            print(f"✅ AI analysis returned {len(matches)} matches")
            
            if matches and matches[0].get("context"):
                context = matches[0]["context"]
                if "AI Analysis:" in context:
                    print("✅ AI enhancement detected in results")
                else:
                    print("ℹ️  AI analysis available but may not have enhanced this result")
            
            return True
        else:
            print(f"❌ AI analysis test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ AI analysis test error: {str(e)}")
        return False

def run_performance_test():
    """Run performance test"""
    print("\n⚡ Performance Test")
    
    query = {
        "query": "Navigation equipment malfunction during voyage",
        "vessel_type": "All",
        "top_k": 5
    }
    
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            response = requests.post(f"{API_BASE_URL}/search", json=query)
            if response.status_code == 200:
                end_time = time.time()
                times.append(end_time - start_time)
            else:
                print(f"❌ Performance test request {i+1} failed")
        except Exception as e:
            print(f"❌ Performance test error: {str(e)}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"✅ Average response time: {avg_time:.3f} seconds")
        print(f"   Min: {min(times):.3f}s, Max: {max(times):.3f}s")
        return True
    else:
        return False

def main():
    """Run all tests"""
    print("🚢 VIQ AI System Test Suite")
    print("=" * 40)
    
    tests = [
        ("API Health", test_api_health),
        ("Vessel Types", test_vessel_types),
        ("System Stats", test_stats),
        ("Search Functionality", test_search),
        ("AI Analysis", test_ai_analysis),
        ("Performance", run_performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())