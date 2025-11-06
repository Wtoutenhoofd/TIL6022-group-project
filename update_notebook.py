import json

# Read the data descriptions
with open('data_descriptions.md', 'r', encoding='utf-8') as f:
    data_content = f.read()

# Remove the markdown title since it will be a notebook section
data_content = data_content.replace('# Data Used in This Project\n\n', '')

# Read the notebook
with open('project_template.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find section 3 "Data used" and update it
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '# 3. Data used' in source:
            # Found it! Now replace with comprehensive content
            new_content = "# 3. Data Used\n\n" + data_content
            # Split into lines for notebook format
            notebook['cells'][i]['source'] = [line + '\n' if not line.endswith('\n') else line 
                                              for line in new_content.split('\n')]
            print(f"Updated cell {i} with data descriptions")
            break

# Save the updated notebook
with open('project_template.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Successfully updated project_template.ipynb with data descriptions!")
