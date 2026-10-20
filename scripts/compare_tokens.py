import sys
import os
import json
import time

# Add project root to sys.path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from toon import encode as toon_encode

# Try to import a tokenizer. If not available, we'll use a rough estimate.
try:
    from transformers import AutoTokenizer
    # GPT2 tokenizer is a good general proxy for BPE token counts
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    def count_tokens(text):
        return len(tokenizer.encode(text))
    tokenizer_name = "GPT-2 (Standard BPE)"
except ImportError:
    # Fallback: Rough estimate (4 chars per token)
    def count_tokens(text):
        return len(text.split()) * 1.3 # Rough word-to-token ratio
    tokenizer_name = "Word-count Estimate"

# Mock Data: 3 Realistic Resumes
mock_resumes = [
    {
        "source": "alice_resume.pdf",
        "name": "Alice Johnson",
        "skills": "Python, Django, FastAPI, Docker, Kubernetes, AWS, PostgreSQL, Redis",
        "experience": "Senior Backend Engineer at TechCorp (5 years). Built scalable microservices.",
        "education": "BS Computer Science, MIT"
    },
    {
        "source": "bob_resume.pdf",
        "name": "Bob Smith",
        "skills": "JavaScript, React, Node.js, TypeScript, MongoDB, GraphQL, AWS Lambda",
        "experience": "Frontend Developer at WebSolutions (3 years). Redesigned main dashboard.",
        "education": "BS Information Technology, GT"
    },
    {
        "source": "charlie_resume.pdf",
        "name": "Charlie Davis",
        "skills": "Java, Spring Boot, Hibernate, MySQL, Kafka, Jenkins, CI/CD",
        "experience": "Software Engineer at EnterpriseDev (4 years). Optimized payment processing.",
        "education": "MS Computer Science, Stanford"
    }
]

def run_comparison():
    print(f"--- Token Comparison Analysis ---")
    print(f"Tokenizer used: {tokenizer_name}")
    print(f"Dataset: {len(mock_resumes)} mock resumes\n")

    # 1. Standard JSON Format
    json_output = json.dumps(mock_resumes, indent=2)
    json_tokens = count_tokens(json_output)
    
    print("Format: JSON (Standard)")
    print("-" * 30)
    print(json_output)
    print("-" * 30)
    print(f"Token Count: {json_tokens}\n")

    # 2. TOON Format
    toon_output = toon_encode({"resumes": mock_resumes})
    toon_tokens = count_tokens(toon_output)
    
    print("Format: TOON (Official Spec v3.0)")
    print("-" * 30)
    print(toon_output)
    print("-" * 30)
    print(f"Token Count: {toon_tokens}\n")

    # 3. Results
    savings = json_tokens - toon_tokens
    percent_savings = (savings / json_tokens) * 100
    
    print(f"--- Summary ---")
    print(f"JSON Tokens: {json_tokens}")
    print(f"TOON Tokens: {toon_tokens}")
    print(f"Savings:     {savings} tokens")
    print(f"Efficiency:  {percent_savings:.2f}% reduction")

if __name__ == "__main__":
    run_comparison()
