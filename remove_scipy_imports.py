#!/usr/bin/env python3
"""
Script to remove scipy imports from all Python files.
"""

import os
import re

def remove_scipy_imports(filepath):
    """Remove scipy imports from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove 'from scipy import' statements
    content = re.sub(r'^from scipy import.*$', '', content, flags=re.MULTILINE)
    
    # Remove 'import scipy' statements
    content = re.sub(r'^import scipy.*$', '', content, flags=re.MULTILINE)
    
    # Remove 'scipy.' usage (we'll handle this case by case)
    # For now, just remove import statements
    
    # Clean up empty lines
    lines = content.split('\n')
    lines = [line for line in lines if line.strip() != '']
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return len(lines)

def main():
    """Main function."""
    root_dir = 'seismo_framework'
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                print(f"Processing {filepath}...")
                try:
                    remove_scipy_imports(filepath)
                except Exception as e:
                    print(f"  Error processing {filepath}: {e}")

if __name__ == "__main__":
    main()
