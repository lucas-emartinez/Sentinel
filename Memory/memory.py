import json
from time import sleep

class MemoryData:
    def __init__(self):
        self.data = {}
        self.load_memory_data()
        
    def get(self, key):
        """Obtiene el valor asociado a una clave."""
        return self.data.get(key)
    
    def set(self, key, value):
        """Establece un valor asociado a una clave."""
        self.data[key] = value
        self.save_memory_data()
    
    def save_memory_data(self):
        """Guarda los datos en un archivo JSON."""
        with open("memory.json", "w") as f:
            json.dump(self.data, f)

    def load_memory_data(self):
        """Carga los datos desde un archivo JSON."""
        try:
            with open("memory.json", "r") as f:
                data = json.load(f)
                self.data = data
        except (FileNotFoundError, json.JSONDecodeError):
            print("No se pudo cargar la memoria. Reintentando...")
            sleep(1)
            self.load_memory_data()
            
    def get_nested(self, keys):
        """Obtiene un valor anidado utilizando una notación de puntos."""
        return self._get_nested_value(keys, self.data)

    def set_nested(self, keys, value):
        """Establece un valor en un diccionario anidado mediante una notación de puntos."""
        self._set_nested_value(keys, value, self.data)

    def _get_nested_value(self, keys, current_data):
        """Obtiene un valor anidado utilizando una notación de puntos."""
        keys = keys.split(".")
        for key in keys:
            current_data = current_data.get(key)  # Cambiado a current_data
            if current_data is None:
                return None
        return current_data

    def _set_nested_value(self, keys, value, current_data):
        """Establece un valor en un diccionario anidado mediante una notación de puntos."""
        print(value)
        keys = keys.split(".")
        for key in keys[:-1]:
            current_data = current_data.setdefault(key, {})  # Cambiado a current_data
        current_data[keys[-1]] = value
        self.save_memory_data()
