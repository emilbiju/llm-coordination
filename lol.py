import sys
print("Python path:", sys.path)
import importlib.util
spec = importlib.util.find_spec('overcooked_ai_py')
print("overcooked_ai_py spec:", spec)