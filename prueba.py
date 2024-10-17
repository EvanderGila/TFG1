import platform

# Obtener información del sistema
system_info = platform.system()
version_info = platform.version()
release_info = platform.release()

print(f"Sistema operativo: {system_info}")
print(f"Versión: {version_info}")
print(f"Liberación: {release_info}")