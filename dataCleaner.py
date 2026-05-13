import json

def clean_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for card_name, details in data.items():
        if not isinstance(details, dict):
            continue
            
        tags_originais = details.get("tags")
        
        if tags_originais is None:
            details["tags"] = []
            continue
            
        # remove tags com palavra "annotation"
        tags_limpas = []
        for tag in tags_originais:
            if isinstance(tag, str):
                if "annotation" not in tag.lower():
                    tags_limpas.append(tag.strip())
        
        # remove duplicatas e ordena alfabeticamente
        details["tags"] = sorted(list(set(tags_limpas)))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"'{output_file}' foi gerado")

if __name__ == "__main__":
    clean_data('card_dataset.json', 'clean_card_dataset.json')