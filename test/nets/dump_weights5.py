import gguf
reader = gguf.GGUFReader("test/big_models/Lite-Oute-1-65M-FP16.gguf")

for field in reader.fields.values():
    if "eps" in field.name.lower():
        print(f"Norm epsilon: {field.name} = {field.parts[field.data[0]]}")
        print(f"Norm epsilon type: {field.types}")
