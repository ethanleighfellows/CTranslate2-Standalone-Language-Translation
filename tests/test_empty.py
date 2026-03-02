import logging
from standalone_translator import translate

logging.basicConfig(level=logging.INFO)

print("Starting standard test...")
print(translate(["Hello, world!", "How are you?"], "pt"))

print("\nStarting empty string test...")
print(translate(["", "Test"], "pt"))

print("\nStarting blank space test...")
print(translate([" ", "Test"], "pt"))

