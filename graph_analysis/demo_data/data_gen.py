import pandas as pd
from faker import Faker
import random

def generate_synthetic_ecommerce_data(num_customers=100, num_products=50, num_purchases=200):
    fake = Faker()

    customers = []
    for i in range(1, num_customers + 1):
        customer_id = f"C_{i}"
        customer_name = fake.name()
        customers.append({'customer_id': customer_id, 'customer_name': customer_name})

    product_names = [
        "Laptop", "Smartphone", "Headphones", "Camera", "Smartwatch", "Tablet", 
        "Monitor", "Keyboard", "Mouse", "Printer", "Speaker", "Router", 
        "External Hard Drive", "USB Flash Drive", "Memory Card", "Webcam", 
        "Microphone", "Projector", "Charger", "Power Bank", "Bluetooth Adapter", 
        "Graphic Card", "Processor", "Motherboard", "RAM", "SSD", "HDD", 
        "Cooling Fan", "Computer Case", "Power Supply", "Cable", "Adapter", 
        "Smart Home Device", "Wearable Tech", "E-Reader", "VR Headset", 
        "Gaming Console", "Game Controller", "Drone", "Action Camera", 
        "Fitness Tracker", "Electric Toothbrush", "Hair Dryer", "Coffee Maker", 
        "Blender", "Air Purifier", "Robot Vacuum", "Instant Pot", "Slow Cooker"
    ]

    num_products = min(num_products, len(product_names))

    products = []
    for i in range(1, num_products + 1):
        product_id = f"P_{i}"
        product_name = product_names[i - 1]
        products.append({'product_id': product_id, 'product_name': product_name})

    purchases = []
    for _ in range(num_purchases):
        customer_id = random.choice(customers)['customer_id']
        product_id = random.choice(products)['product_id']
        purchase_date = fake.date_this_year()
        purchases.append({'customer_id': customer_id, 'product_id': product_id, 'purchase_date': purchase_date})

    co_purchases = []
    for product in products:
        if random.random() > 0.5:
            other_product = random.choice(products)
            if product['product_id'] != other_product['product_id']:
                co_purchases.append({'product_id_1': product['product_id'], 'product_id_2': other_product['product_id']})

    return customers, products, purchases, co_purchases

def save_to_csv(customers, products, purchases, co_purchases):
    customers_df = pd.DataFrame(customers)
    products_df = pd.DataFrame(products)
    purchases_df = pd.DataFrame(purchases)
    co_purchases_df = pd.DataFrame(co_purchases)

    customers_df.to_csv('demo_data/customers.csv', index=False)
    products_df.to_csv('demo_data/products.csv', index=False)
    purchases_df.to_csv('demo_data/purchases.csv', index=False)
    co_purchases_df.to_csv('demo_data/co_purchases.csv', index=False)
    
customers, products, purchases, co_purchases = generate_synthetic_ecommerce_data()

save_to_csv(customers, products, purchases, co_purchases)
