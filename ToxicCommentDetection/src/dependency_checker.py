import os
import ast
import sys

def extract_imports(filepath):
    """Extrae los nombres de los paquetes importados en un archivo .py"""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
    except Exception as e:
        print(f"Error al analizar {filepath}: {e}", file=sys.stderr)
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Solo el nombre base del paquete (p.ej. 'pandas' en lugar de 'pandas.DataFrame')
                package_name = alias.name.split('.')[0]
                imports.add(package_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # node.module puede ser None en casos raros
                package_name = node.module.split('.')[0]
                imports.add(package_name)
    return imports

def main():
    # Buscar todos los archivos .py en la carpeta actual y subcarpetas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_imports = set()
    
    print(f"Analizando archivos en: {current_dir}")
    
    for root, _, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                print(f"Procesando: {path}")
                imports = extract_imports(path)
                all_imports.update(imports)
    
    # Filtrar módulos estándar (no necesitan instalación)
    stdlib_modules = set(sys.builtin_module_names)
    external_packages = sorted([pkg for pkg in all_imports if pkg not in stdlib_modules])
    
    print("\nPaquetes externos utilizados:")
    for package in external_packages:
        print(package)

if __name__ == '__main__':
    main()