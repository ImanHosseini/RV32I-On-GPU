import sys
import os
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

def get_start_file_offset(elf_file):
    # Get the section containing _start
    text_section = elf_file.get_section_by_name('.cuda_constant')
    
    # Get the symbol table
    symtab = elf_file.get_section_by_name('.symtab')
    
    for symbol in symtab.iter_symbols():
        if symbol.name == '_start':
            return text_section['sh_offset'] + (symbol['st_value'] - text_section['sh_addr'])
    return None

def patch_instruction(instruction: int, new_imm: int) -> int:
    # The destination register and the opcode are in the lower 12 bits
    lower_12_bits = instruction & 0xFFF
    # Construct the new instruction with the provided immediate value (bits 12:31)
    return (new_imm << 12) | lower_12_bits

def patch_elf(input_filename, output_filename):
    # First, let's find the file offset of the _start symbol
    with open(input_filename, 'rb') as f:
        elf_file = ELFFile(f)
        start_file_offset = get_start_file_offset(elf_file)

    if start_file_offset is None:
        print("Error: _start not found in the ELF file.")
        return

    # Derive the offsets for the AUIPC instructions
    auipc_a0_offset = start_file_offset + 0x8
    auipc_a2_offset = start_file_offset + 0x10

    # Load the entire file
    with open(input_filename, 'rb') as f:
        data = bytearray(f.read())

    # Patch the offsets
    auipc_a0_instruction = int.from_bytes(data[auipc_a0_offset:auipc_a0_offset+4], byteorder='little')
    auipc_a2_instruction = int.from_bytes(data[auipc_a2_offset:auipc_a2_offset+4], byteorder='little')

    patched_auipc_a0 = patch_instruction(auipc_a0_instruction, 0)
    patched_auipc_a2 = patch_instruction(auipc_a2_instruction, 0)

    data[auipc_a0_offset:auipc_a0_offset+4] = patched_auipc_a0.to_bytes(4, byteorder='little')
    data[auipc_a2_offset:auipc_a2_offset+4] = patched_auipc_a2.to_bytes(4, byteorder='little')

    # Save the modified ELF file
    with open(output_filename, 'wb') as out_f:
        out_f.write(data)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 script_name.py <input_elf> [-replace]")
        sys.exit(1)

    input_file = sys.argv[1]
    
    directory, filename = os.path.split(input_file)

    if "-replace" in sys.argv:
        output_file = input_file
    else:
        output_file = os.path.join(directory, 'patched_' + filename)

    patch_elf(input_file, output_file)
    print(f"Patched ELF written to {output_file}")
