#!/usr/bin/env python3
"""
Check if your implementation has the keyword argument bug.
Run: python check_implementation.py
"""

from pathlib import Path
import re


def check_file(filepath):
    """Check a single file for FakeQuantOp keyword argument usage"""
    try:
        content = Path(filepath).read_text()
    except FileNotFoundError:
        return None, f"File not found: {filepath}"
    
    issues = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # Look for FakeQuantOp.apply with keyword arguments
        if 'FakeQuantOp.apply' in line and 'num_bits=' in line:
            # Extract the actual call
            match = re.search(r'FakeQuantOp\.apply\([^)]+\)', line)
            if match:
                issues.append((i, line.strip(), match.group(0)))
    
    return issues, None


def main():
    print("="*70)
    print("CHECKING IMPLEMENTATION FOR KEYWORD ARGUMENT BUGS")
    print("="*70)
    print()
    
    files_to_check = [
        'src/quant_lens/quantization.py',
        'src/quant_lens/geometry.py',
        'src/quant_lens/hessian.py',
        'src/quant_lens/core.py',
        'src/quant_lens/plotting.py',
    ]
    
    total_issues = 0
    
    for filepath in files_to_check:
        issues, error = check_file(filepath)
        
        if error:
            print(f"⚠️  {filepath}")
            print(f"   {error}")
            continue
        
        if issues:
            print(f"❌ {filepath} - ISSUES FOUND:")
            for line_num, line_content, call in issues:
                print(f"   Line {line_num}: {call}")
                print(f"   Full line: {line_content}")
                # Suggest fix
                fixed_call = call.replace('num_bits=', '')
                print(f"   Fix: {fixed_call}")
                print()
                total_issues += 1
        else:
            print(f"✅ {filepath} - OK")
    
    print()
    print("="*70)
    
    if total_issues > 0:
        print(f"⚠️  FOUND {total_issues} ISSUE(S)")
        print()
        print("How to fix:")
        print("  1. Open the file(s) listed above")
        print("  2. Find the line number indicated")
        print("  3. Remove 'num_bits=' from the FakeQuantOp.apply() call")
        print("  4. Example:")
        print("     Before: FakeQuantOp.apply(x, num_bits=8)")
        print("     After:  FakeQuantOp.apply(x, 8)")
        print()
        print("  Or run this one-liner:")
        print()
        print("  python3 -c \"")
        print("  from pathlib import Path")
        print("  p = Path('src/quant_lens/quantization.py')")
        print("  content = p.read_text()")
        print("  content = content.replace('FakeQuantOp.apply(self.weight, num_bits=self.num_bits)', 'FakeQuantOp.apply(self.weight, self.num_bits)')")
        print("  p.write_text(content)")
        print("  print('Fixed!')\"")
    else:
        print("✅ NO ISSUES FOUND!")
        print()
        print("Your implementation is correct. All FakeQuantOp.apply() calls")
        print("use positional arguments as required by PyTorch.")
    
    print("="*70)
    
    return total_issues


if __name__ == "__main__":
    exit(main())
