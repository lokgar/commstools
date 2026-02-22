def replace_in_file(filename, old_str, new_str):
    with open(filename, 'r') as f:
        content = f.read()
    content = content.replace(old_str, new_str)
    with open(filename, 'w') as f:
        f.write(content)

replace_in_file('tests/test_equalizers.py', 'step_size=0.001,', 'step_size=0.003,')
replace_in_file('tests/test_equalizers.py', 'n_symbols = 5000', 'n_symbols = 3000')
