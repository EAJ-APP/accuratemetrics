"""
Script para recuperar archivos DOCX corruptos
Uso en Google Colab
"""

import zipfile
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

def recover_corrupted_docx(input_file, output_file=None):
    """
    Intenta recuperar un archivo DOCX corrupto

    Args:
        input_file: Ruta al archivo DOCX corrupto
        output_file: Ruta para guardar el archivo recuperado (opcional)

    Returns:
        bool: True si la recuperación fue exitosa
    """
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_recovered.docx"

    temp_dir = "temp_docx_extraction"

    try:
        # Método 1: Intentar reparar el ZIP corrupto
        print("Método 1: Intentando reparar estructura ZIP...")
        if repair_zip_structure(input_file, output_file):
            print(f"✓ Archivo recuperado exitosamente: {output_file}")
            return True

        # Método 2: Extraer y reconstruir
        print("\nMétodo 2: Extrayendo y reconstruyendo...")
        os.makedirs(temp_dir, exist_ok=True)

        # Intentar extraer con diferentes métodos
        extracted = False

        # Intento 1: Extracción normal
        try:
            with zipfile.ZipFile(input_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            extracted = True
            print("✓ Extracción exitosa")
        except zipfile.BadZipFile:
            print("✗ Error en extracción normal, intentando modo permisivo...")

            # Intento 2: Extracción con allowZip64
            try:
                with zipfile.ZipFile(input_file, 'r', allowZip64=True) as zip_ref:
                    for item in zip_ref.namelist():
                        try:
                            zip_ref.extract(item, temp_dir)
                        except:
                            print(f"  - No se pudo extraer: {item}")
                extracted = True
                print("✓ Extracción parcial exitosa")
            except Exception as e:
                print(f"✗ Error en extracción: {e}")

        if extracted:
            # Limpiar y reparar XMLs
            print("\nReparando archivos XML...")
            fix_xml_files(temp_dir)

            # Recrear el DOCX
            print("\nReconstruyendo archivo DOCX...")
            create_docx_from_folder(temp_dir, output_file)
            print(f"✓ Archivo reconstruido: {output_file}")
            return True

        # Método 3: Extracción de texto plano
        print("\nMétodo 3: Extrayendo solo el texto...")
        if extract_text_only(input_file, output_file):
            print(f"✓ Texto extraído y guardado: {output_file}")
            return True

        print("\n✗ No se pudo recuperar el archivo")
        return False

    except Exception as e:
        print(f"✗ Error durante la recuperación: {e}")
        return False

    finally:
        # Limpiar archivos temporales
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def repair_zip_structure(input_file, output_file):
    """Intenta reparar la estructura ZIP del archivo"""
    try:
        # Leer el archivo binario
        with open(input_file, 'rb') as f:
            data = f.read()

        # Buscar el inicio del ZIP (PK signature)
        pk_start = data.find(b'PK\x03\x04')
        if pk_start == -1:
            return False

        # Si hay basura antes del inicio del ZIP, removerla
        if pk_start > 0:
            data = data[pk_start:]
            print(f"  - Removidos {pk_start} bytes de basura al inicio")

        # Buscar el final del ZIP
        pk_end = data.rfind(b'PK\x05\x06')
        if pk_end != -1:
            # Encontrar el final real del central directory
            end_data = data[pk_end:]
            if len(end_data) >= 22:  # Tamaño mínimo del End of Central Directory
                data = data[:pk_end + 22]

        # Guardar el archivo reparado
        with open(output_file, 'wb') as f:
            f.write(data)

        # Verificar si es un ZIP válido ahora
        try:
            with zipfile.ZipFile(output_file, 'r') as zf:
                zf.testzip()
            return True
        except:
            os.remove(output_file)
            return False

    except Exception as e:
        print(f"  - Error en reparación ZIP: {e}")
        return False


def fix_xml_files(directory):
    """Repara archivos XML corruptos"""
    xml_files = Path(directory).rglob('*.xml')

    for xml_file in xml_files:
        try:
            # Leer el contenido
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Intentar parsear
            try:
                ET.fromstring(content)
            except ET.ParseError:
                # Intentar reparar
                content = repair_xml_content(content)
                with open(xml_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  - Reparado: {xml_file.name}")
        except Exception as e:
            print(f"  - No se pudo reparar {xml_file.name}: {e}")


def repair_xml_content(content):
    """Intenta reparar contenido XML corrupto"""
    # Remover caracteres no válidos
    content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')

    # Asegurar que tiene declaración XML
    if not content.strip().startswith('<?xml'):
        content = '<?xml version="1.0" encoding="UTF-8"?>\n' + content

    return content


def create_docx_from_folder(folder, output_file):
    """Crea un archivo DOCX desde una carpeta extraída"""
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as docx:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder)
                docx.write(file_path, arcname)


def extract_text_only(input_file, output_file):
    """Extrae solo el texto del DOCX corrupto"""
    try:
        import io

        with open(input_file, 'rb') as f:
            data = f.read()

        # Buscar contenido de texto en el binario
        text_parts = []

        # Buscar tags de texto XML comunes en DOCX
        patterns = [b'<w:t>', b'<w:t ']

        for pattern in patterns:
            start = 0
            while True:
                pos = data.find(pattern, start)
                if pos == -1:
                    break

                # Buscar el cierre del tag
                end_tag = data.find(b'</w:t>', pos)
                if end_tag != -1:
                    # Extraer el texto
                    text_start = data.find(b'>', pos) + 1
                    text = data[text_start:end_tag].decode('utf-8', errors='ignore')
                    if text.strip():
                        text_parts.append(text)

                start = pos + len(pattern)

        if text_parts:
            # Crear un nuevo DOCX con el texto recuperado
            recovered_text = '\n'.join(text_parts)

            # Guardar como archivo de texto
            text_output = output_file.replace('.docx', '_recovered.txt')
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write(recovered_text)

            print(f"  - Texto extraído: {len(text_parts)} fragmentos")
            return True

        return False

    except Exception as e:
        print(f"  - Error extrayendo texto: {e}")
        return False


# Código para usar en Google Colab
def main_colab():
    """
    Función principal para Google Colab
    """
    print("=" * 60)
    print("RECUPERADOR DE ARCHIVOS DOCX CORRUPTOS")
    print("=" * 60)

    # Instrucciones
    print("\n📋 INSTRUCCIONES:")
    print("1. Sube tu archivo DOCX corrupto usando el botón de archivos")
    print("2. Ingresa el nombre del archivo cuando se solicite")
    print("3. El script intentará recuperar el archivo\n")

    # Obtener nombre del archivo
    filename = input("Ingresa el nombre del archivo DOCX corrupto: ").strip()

    if not os.path.exists(filename):
        print(f"\n✗ Error: No se encontró el archivo '{filename}'")
        print("Asegúrate de haber subido el archivo primero.")
        return

    print(f"\n🔧 Procesando: {filename}")
    print("-" * 60)

    # Intentar recuperar
    success = recover_corrupted_docx(filename)

    print("-" * 60)
    if success:
        print("\n✅ PROCESO COMPLETADO")
        print("Revisa los archivos generados en el panel de archivos.")
    else:
        print("\n⚠️ No se pudo recuperar completamente el archivo")
        print("Verifica si se generó algún archivo de texto con contenido parcial.")


if __name__ == "__main__":
    # Para ejecutar en Google Colab
    main_colab()
