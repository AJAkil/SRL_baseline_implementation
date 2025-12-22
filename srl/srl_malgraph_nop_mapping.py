#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRL + MalGraph Semantic NOP Mapping

This module maps MalGuise's semantic NOPs to MalGraph's 11-dimensional 
block feature increments.

MalGraph Block Features (11-dim):
    [numNc, numSc, numAs, numCalls, numIns, numLIs, numTIs, numCmpIs, numMovIs, numTermIs, numDefIs]
    [0]     [1]    [2]    [3]       [4]     [5]    [6]    [7]      [8]      [9]       [10]

Author: Md Ajwad Akil
Date: December 2025
"""

import re
from typing import List, Dict, Tuple


class SemanticNOPMapper:
    """
    Maps semantic NOP assembly strings to MalGraph feature increments.
    
    Feature indices:
        0: numNc  - Number of numeric constants
        1: numSc  - Number of string constants
        2: numAs  - Number of arithmetic instructions
        3: numCalls - Number of call instructions
        4: numIns - Total number of instructions
        5: numLIs - Number of logic instructions
        6: numTIs - Number of transfer instructions
        7: numCmpIs - Number of compare instructions
        8: numMovIs - Number of move instructions
        9: numTermIs - Number of termination instructions
        10: numDefIs - Number of data definition instructions
    """
    
    # Opcode categories from MalGraph's graph_analysis_ida.py
    ARITHMETIC_OPS = {'add', 'sub', 'div', 'imul', 'idiv', 'mul', 'shl', 'dec', 'inc',
                      'addu', 'addi', 'addiu', 'mult', 'multu', 'divu'}
    
    LOGIC_OPS = {'and', 'andn', 'andnpd', 'andpd', 'andps', 'andnps', 'test', 'xor', 
                 'xorpd', 'pslld', 'andi', 'or', 'ori', 'nor', 'slt', 'slti', 'sltu'}
    
    TRANSFER_OPS = {'jmp', 'jz', 'jnz', 'js', 'je', 'jne', 'jg', 'jle', 'jge', 'ja', 
                    'jnc', 'jb', 'jl', 'jnb', 'jno', 'jnp', 'jns', 'jo', 'jp',
                    'loop', 'loope', 'loopne', 'loopw', 'loopwe', 'loopwne',
                    'beq', 'bne', 'bgtz', 'bltz', 'bgez', 'blez', 'j', 'jal', 'jr', 'jalr'}
    
    COMPARE_OPS = {
        'cmp', 'cmpeqps', 'cmpeqsd', 'cmpeqss', 'cmpleps',
        'cmplesd', 'cmpltpd', 'cmpltps', 'cmpltsd', 'cmpneqpd',
        'cmpneqps', 'cmpnlepd', 'cmpnlesd', 'cmpps', 'cmps',
        'cmpsb', 'cmpsd', 'cmpsw', 'cmpxchg', 'comisd',
        'comiss',
        'cmpeqpd', 'cmpltss', 'cmpnleps', 'cmpnless',
        'cmpnltpd', 'cmpnltps', 'cmpnltsd', 'cmpnltss',
        'cmpunordpd', 'cmpunordps',
        'fcom', 'fcomi', 'fcomip', 'fcomp', 'fcompp', 'ficom', 'ficomp',
        'fucom', 'fucomi', 'fucomip', 'fucomp', 'fucompp',
        'pcmpeqb', 'pcmpeqd', 'pcmpeqw', 'pcmpgtb',
        'pcmpgtd', 'pcmpgtw', 'pfcmpeq', 'pfcmpge', 'pfcmpgt',
        'ucomisd', 'ucomiss',
        'vpcmpeqb', 'vpcmpeqd',
        'vpcmpeqw', 'vpcmpgtb', 'vpcmpgtd', 'vpcmpgtw', 'vucomiss',
        'vcmpsd', 'vcomiss', 'vucomisd',
        }
    
    MOVE_OPS = { 'cmova', 'cmovb', 'cmovbe', 'cmovg', 'cmovge',
                'cmovl', 'cmovle', 'cmovnb', 'cmovno', 'cmovnp',
                'cmovns', 'cmovnz', 'cmovo', 'cmovp', 'cmovs', 'cmovz',
                'fcmovb', 'fcmovbe', 'fcmove', 'fcmovnb',
                'fcmovnbe', 'fcmovne', 'fcmovnu', 'fcmovu',
                'mov', 'movapd', 'movaps', 'movd', 'movdqa', 'movdqu',
                'movhlps', 'movhpd', 'movhps', 'movlhps', 'movlpd',
                'movlps', 'movmskpd', 'movmskps', 'movntdq', 'movnti',
                'movntps', 'movntq', 'movq', 'movs', 'movsb', 'movsd',
                'movss', 'movsw', 'movsx', 'movups', 'movzx',
                'movntpd', 'movupd',
                'pmovmskb', 'pmovzxbd', 'pmovzxwd',
                'vmovapd', 'vmovaps', 'vmovd',
                'vmovddup', 'vmovdqa', 'vmovdqu', 'vmovhps', 'vmovlhps',
                'vmovntdq', 'vmovntpd', 'vmovntps', 'vmovntsd',
                'vmovsd', 'vmovsldup', 'vmovss', 'vmovupd', 'vmovups',
                'vmovhlps', 'vmovlps', 'vmovq', 'vmovshdup'}
    
    TERMINATION_OPS = {'end', 'iret', 'iretw', 'retf', 'reti', 'retfw', 'retn', 
                       'retnw', 'sysexit', 'sysret', 'xabort', 'ret'}
    
    DATA_DEF_OPS = {'dd', 'db', 'dw', 'dq', 'dt', 'extrn', 'unicode'}
    
    CALL_OPS = {'call', 'jal', 'jalr'}
    
    def __init__(self):
        """Initialize the NOP mapper."""
        self.nop_cache = {}
    
    def parse_nop_string(self, nop_str: str) -> List[Tuple[str, str, str]]:
        """
        Parse semantic NOP string into list of (opcode, operand1, operand2).
        
        Args:
            nop_str: Assembly string like "push eax\npop eax\n"
        
        Returns:
            List of tuples: [(opcode, op1, op2), ...]
        
        Example:
            "push eax\npop eax\n" -> [('push', 'eax', ''), ('pop', 'eax', '')]
            "add eax, 0x89h\n" -> [('add', 'eax', '0x89h')]
        """
        instructions = []
        lines = nop_str.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace
            parts = line.split(None, 1)  # Split at most once
            if len(parts) == 0:
                continue
            
            opcode = parts[0].lower()
            
            if len(parts) == 1:
                # No operands (e.g., "nop")
                instructions.append((opcode, '', ''))
            else:
                # Has operands
                operands = parts[1].split(',')
                op1 = operands[0].strip() if len(operands) > 0 else ''
                op2 = operands[1].strip() if len(operands) > 1 else ''
                instructions.append((opcode, op1, op2))
        
        return instructions
    
    def has_numeric_constant(self, operand: str) -> bool:
        """
        Check if operand contains a numeric constant.
        
        Args:
            operand: Operand string like "0x89h", "100", "[eax+4]"
        
        Returns:
            True if contains numeric constant
        """
        # Hexadecimal patterns: 0x89, 0x89h, 89h
        if re.search(r'0x[0-9a-fA-F]+h?|[0-9a-fA-F]+h', operand):
            return True
        # Decimal patterns: 123, [eax+4]
        if re.search(r'\d+', operand):
            return True
        return False
    
    def compute_feature_increment(self, nop_str: str) -> List[int]:
        """
        Compute 11-dimensional feature increment for a semantic NOP.
        
        Args:
            nop_str: Assembly string like "push eax\npop eax\n"
        
        Returns:
            Feature increment array [numNc, numSc, numAs, numCalls, numIns, 
                                     numLIs, numTIs, numCmpIs, numMovIs, 
                                     numTermIs, numDefIs]
        
        Example:
            "push eax\npop eax\n" -> [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2]
        """
        # Check cache
        if nop_str in self.nop_cache:
            return self.nop_cache[nop_str].copy()
        
        # Initialize feature array
        features = [0] * 11
        # [numNc, numSc, numAs, numCalls, numIns, numLIs, numTIs, numCmpIs, 
        #  numMovIs, numTermIs, numDefIs]
        
        instructions = self.parse_nop_string(nop_str)
        
        for opcode, op1, op2 in instructions:
            # Increment total instruction count
            features[4] += 1  # numIns
            
            # Check for numeric constants in operands
            if self.has_numeric_constant(op1) or self.has_numeric_constant(op2):
                features[0] += 1  # numNc
            
            # Categorize by opcode
            if opcode in self.CALL_OPS:
                features[3] += 1  # numCalls
            
            if opcode in self.ARITHMETIC_OPS:
                features[2] += 1  # numAs
            
            if opcode in self.LOGIC_OPS:
                features[5] += 1  # numLIs
            
            if opcode in self.TRANSFER_OPS:
                features[6] += 1  # numTIs
            
            if opcode in self.COMPARE_OPS:
                features[7] += 1  # numCmpIs
            
            if opcode in self.MOVE_OPS:
                features[8] += 1  # numMovIs
            
            if opcode in self.TERMINATION_OPS:
                features[9] += 1  # numTermIs
            
            if opcode in self.DATA_DEF_OPS:
                features[10] += 1  # numDefIs
        
        # Cache result
        self.nop_cache[nop_str] = features.copy()
        
        return features
    
    def apply_nop_to_block_features(self, block_features: List[int], nop_str: str) -> List[int]:
        """
        Apply semantic NOP increment to existing block features (IN-PLACE).
        
        This modifies the block_features list directly by adding the NOP's 
        feature increments. Use this when injecting NOPs into existing basic blocks.
        
        Args:
            block_features: Existing 11-dim feature vector [numNc, numSc, numAs, ...]
            nop_str: Assembly string like "push eax\npop eax\n"
        
        Returns:
            Modified block_features (same list, modified in-place)
        
        Example:
            >>> block = [5, 2, 3, 1, 10, 2, 1, 0, 4, 0, 0]  # Original block
            >>> mapper.apply_nop_to_block_features(block, "push eax\npop eax\n")
            >>> print(block)  # [5, 2, 3, 1, 12, 2, 1, 0, 4, 0, 2]  # Updated!
        """
        nop_increment = self.compute_feature_increment(nop_str)
        
        # Add increment to existing features
        for i in range(11):
            block_features[i] += nop_increment[i]
        
        return block_features
    
    def apply_multiple_nops_to_block(self, block_features: List[int], 
                                     nop_list: List[str]) -> List[int]:
        """
        Apply multiple semantic NOPs to existing block features (IN-PLACE).
        
        Args:
            block_features: Existing 11-dim feature vector
            nop_list: List of NOP assembly strings
        
        Returns:
            Modified block_features (same list, modified in-place)
        
        Example:
            >>> block = [5, 2, 3, 1, 10, 2, 1, 0, 4, 0, 0]
            >>> nops = ["xor eax, eax\n", "mov ebx, ebx\n", "test ecx, ecx\n"]
            >>> mapper.apply_multiple_nops_to_block(block, nops)
            >>> print(block)  # Features incremented by sum of all NOPs
        """
        for nop_str in nop_list:
            self.apply_nop_to_block_features(block_features, nop_str)
        
        return block_features
    
    def generate_malguise_nop_list(self) -> List[Dict[str, any]]:
        """
        Generate full list of MalGuise semantic NOPs with feature increments.
        
        Returns:
            List of dicts with 'nop_str' and 'features' keys
        """
        nop_list = []
        
        # Register lists from MalGuise
        all_reg_list = ['eax', 'ebx', 'ecx', 'edx', 'ch', 'cl', 'ax', 'bx', 
                        'cx', 'dx', 'ah', 'al', 'bh', 'bl', 'dh', 'dl']
        reg_list_32_16 = ['eax', 'ebx', 'ecx', 'edx', 'ax', 'bx', 'cx', 'dx']
        reg_list_32 = ['eax', 'ebx', 'ecx', 'edx']
        
        # Instruction templates
        ins_all = [
            'mov reg, reg\n',
            'dec reg\ninc reg\n',
            'xchg reg, reg\n',
            'test reg, reg\n',
            'cmp reg, reg\n',
            'or reg, reg\n',
            'xor reg, 0xccefh\nxor reg, 0xccefh\n',
            'add reg, 0x89h\nsub reg, 0x89h\n',
            'xor reg, 0x89h\nxor reg, 0x89h\n',
            'and reg, reg\n',
            'add reg, 0xccefh\nsub reg, 0xccefh\n'
        ]
        ins_32_16 = ['push reg\npop reg\n']
        ins_32 = ['bswap reg\nbswap reg\n']
        
        # Generate all combinations
        for reg in all_reg_list:
            for ins in ins_all:
                nop_str = ins.replace('reg', reg)
                features = self.compute_feature_increment(nop_str)
                nop_list.append({
                    'nop_str': nop_str,
                    'features': features,
                    'register': reg,
                    'template': ins
                })
        
        for reg in reg_list_32_16:
            for ins in ins_32_16:
                nop_str = ins.replace('reg', reg)
                features = self.compute_feature_increment(nop_str)
                nop_list.append({
                    'nop_str': nop_str,
                    'features': features,
                    'register': reg,
                    'template': ins
                })
        
        for reg in reg_list_32:
            for ins in ins_32:
                nop_str = ins.replace('reg', reg)
                features = self.compute_feature_increment(nop_str)
                nop_list.append({
                    'nop_str': nop_str,
                    'features': features,
                    'register': reg,
                    'template': ins
                })
        
        return nop_list
    
    def print_nop_statistics(self):
        """Print statistics about NOP feature impacts."""
        nop_list = self.generate_malguise_nop_list()
        
        print(f"Total NOPs: {len(nop_list)}")
        print("\nSample NOPs with feature impacts:")
        print("-" * 80)
        
        # Show diverse examples
        examples = [
            nop_list[0],   # mov
            nop_list[16],  # dec/inc
            nop_list[32],  # xchg
            nop_list[48],  # test
            next((n for n in nop_list if 'push' in n['nop_str']), None),
            next((n for n in nop_list if 'bswap' in n['nop_str']), None),
        ]
        
        for nop_data in examples:
            if nop_data is None:
                continue
            print(f"\nNOP: {nop_data['nop_str'][:50]}...")
            print(f"Features: {nop_data['features']}")
            print(f"  numIns: {nop_data['features'][4]}")
            print(f"  numAs: {nop_data['features'][2]}")
            print(f"  numLIs: {nop_data['features'][5]}")
            print(f"  numMovIs: {nop_data['features'][8]}")
            print(f"  numDefIs: {nop_data['features'][10]}")


# Example usage
if __name__ == "__main__":
    mapper = SemanticNOPMapper()
    
    # Test individual NOPs
    test_nops = [
        "mov eax, eax\n",
        "add eax, 0x89h\nsub eax, 0x89h\n",
        "xor eax, eax\n",
        "test eax, eax\n",
    ]
    
    print("Testing individual NOPs (compute increment):")
    print("=" * 80)
    for nop in test_nops:
        features = mapper.compute_feature_increment(nop)
        print(f"\nNOP: {nop.strip()}")
        print(f"Increment: {features}")
        print(f"  [numNc, numSc, numAs, numCalls, numIns, numLIs, numTIs, "
              f"numCmpIs, numMovIs, numTermIs, numDefIs]")
    
    print("\n" + "=" * 80)
    print("\nTesting in-place block feature update:")
    print("=" * 80)
    
    # Simulate existing basic block features
    original_block = [5, 2, 3, 1, 10, 2, 1, 0, 4, 0, 0]
    print(f"\n📦 Original Block Features: {original_block}")
    print(f"   [numNc={original_block[0]}, numSc={original_block[1]}, "
          f"numAs={original_block[2]}, numCalls={original_block[3]}, "
          f"numIns={original_block[4]}, ...]")
    
    # Inject a single NOP
    nop_to_inject = "push eax\npop eax\n"
    print(f"\n💉 Injecting NOP: {nop_to_inject.strip()}")
    nop_increment = mapper.compute_feature_increment(nop_to_inject)
    print(f"   NOP Increment: {nop_increment}")
    
    # Apply NOP to block (in-place modification)
    updated_block = original_block.copy()  # Copy for demo
    mapper.apply_nop_to_block_features(updated_block, nop_to_inject)
    print(f"\n✅ Updated Block Features: {updated_block}")
    print(f"   [numNc={updated_block[0]}, numSc={updated_block[1]}, "
          f"numAs={updated_block[2]}, numCalls={updated_block[3]}, "
          f"numIns={updated_block[4]}, ...]")
    print(f"\n   Δ Change: numIns +{updated_block[4] - original_block[4]}, "
          f"numDefIs +{updated_block[10] - original_block[10]}")
    
    # Inject multiple NOPs
    print("\n" + "=" * 80)
    print("Testing multiple NOP injection:")
    print("=" * 80)
    
    multi_block = [5, 2, 3, 1, 10, 2, 1, 0, 4, 0, 0]
    nops_to_inject = [
        "push eax\npop eax\n",
        "mov ebx, ebx\n",
        "test ecx, ecx\n"
    ]
    
    print(f"\n📦 Original Block: {multi_block}")
    print(f"💉 Injecting {len(nops_to_inject)} NOPs:")
    for nop in nops_to_inject:
        print(f"   - {nop.strip()}")
    
    mapper.apply_multiple_nops_to_block(multi_block, nops_to_inject)
    print(f"\n✅ Updated Block: {multi_block}")
    print(f"   Total instructions increased by: {multi_block[4] - 10}")
    
    print("\n" + "=" * 80)
    print("\nFull MalGuise NOP List Statistics:")
    print("=" * 80)
    mapper.print_nop_statistics()
