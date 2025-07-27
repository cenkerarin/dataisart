#!/usr/bin/env python3
"""
Security Check Script
====================

Checks the project for potential security issues before committing to version control.
Run this before pushing to GitHub.
"""

import os
import re
import sys
from pathlib import Path

def check_for_secrets():
    """Check for hardcoded secrets in the codebase."""
    secret_patterns = [
        r'sk-proj-[A-Za-z0-9-_]{40,}',  # OpenAI API keys
        r'api_key\s*=\s*["\'][^"\']+["\']',  # Hardcoded API keys
        r'password\s*=\s*["\'][^"\']+["\']',  # Hardcoded passwords
        r'secret\s*=\s*["\'][^"\']+["\']',  # Hardcoded secrets
        r'token\s*=\s*["\'][A-Za-z0-9-_]{20,}["\']',  # Hardcoded tokens
    ]
    
    issues = []
    
    # Files to check
    for file_path in Path('.').rglob('*.py'):
        if 'venv' in str(file_path) or '__pycache__' in str(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in secret_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(f"{file_path}:{i} - Potential secret detected")
        except Exception:
            continue
    
    return issues

def check_gitignore():
    """Check if .gitignore contains necessary security patterns."""
    required_patterns = [
        '.env',
        '*.env',
        '*api_key*',
        '*secret*',
        '*password*',
        '*credentials*',
    ]
    
    try:
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
    except FileNotFoundError:
        return ["No .gitignore file found"]
    
    missing_patterns = []
    for pattern in required_patterns:
        if pattern not in gitignore_content:
            missing_patterns.append(f"Missing pattern: {pattern}")
    
    return missing_patterns

def check_env_files():
    """Check for .env files that shouldn't be committed."""
    env_files = []
    
    for file_path in Path('.').rglob('.env*'):
        if file_path.name != '.env.template' and file_path.name != 'env.template':
            env_files.append(str(file_path))
    
    return env_files

def check_config_files():
    """Check for config files with potential secrets."""
    dangerous_files = []
    
    for file_path in Path('.').rglob('*config*.json'):
        if 'template' not in str(file_path):
            dangerous_files.append(str(file_path))
    
    return dangerous_files

def main():
    """Run all security checks."""
    print("🔒 Running Security Check...")
    print("=" * 40)
    
    all_good = True
    
    # Check 1: Hardcoded secrets
    print("1. Checking for hardcoded secrets...")
    secrets = check_for_secrets()
    if secrets:
        print("❌ Found potential secrets:")
        for secret in secrets:
            print(f"   {secret}")
        all_good = False
    else:
        print("✅ No hardcoded secrets found")
    
    # Check 2: .gitignore
    print("\n2. Checking .gitignore...")
    missing_patterns = check_gitignore()
    if missing_patterns:
        print("⚠️  Missing .gitignore patterns:")
        for pattern in missing_patterns:
            print(f"   {pattern}")
        all_good = False
    else:
        print("✅ .gitignore looks good")
    
    # Check 3: Environment files
    print("\n3. Checking for .env files...")
    env_files = check_env_files()
    if env_files:
        print("❌ Found .env files that shouldn't be committed:")
        for env_file in env_files:
            print(f"   {env_file}")
        all_good = False
    else:
        print("✅ No problematic .env files found")
    
    # Check 4: Config files
    print("\n4. Checking for config files...")
    config_files = check_config_files()
    if config_files:
        print("⚠️  Found config files (verify they don't contain secrets):")
        for config_file in config_files:
            print(f"   {config_file}")
    else:
        print("✅ No suspicious config files found")
    
    # Summary
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 Security check passed! Safe to commit.")
        print("\n💡 Remember:")
        print("   - Use environment variables for API keys")
        print("   - Copy env.template to .env for local development")
        print("   - Never commit .env files")
    else:
        print("⚠️  Security issues found! Fix before committing.")
        print("\n🔧 To fix:")
        print("   - Remove hardcoded secrets")
        print("   - Use environment variables")
        print("   - Add files to .gitignore")
        sys.exit(1)

if __name__ == "__main__":
    main() 