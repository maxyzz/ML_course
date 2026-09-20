from datetime import datetime


class CustomerDataset:
    def __init__(self, customers_data, source):
        # last_update не передаем, его будет считать класс самостоятельно
        self.customers_data = customers_data
        self.source = source
        self.last_update = datetime.now()

    def display_info(self):
        print(f"Источник: {self.source}, Последнее обновление: {self.last_update}, Количество покупателей: {len(self.customers_data)}")

    def filter_customers(self, min_age):
        return [c for c in self.customers_data if int(c.get("age", 0) or 0) >= min_age]
    
    def add_customer(self,name, age):
        self.customers_data.append({"name": name, "age": age})
        self.last_update = datetime.now()
        return self.customers_data

    def update_data(self,new_data):
        self.customers_data = new_data
        self.last_update = datetime.now()
        return self.customers_data

class PremiumCustomerDataset(CustomerDataset):
    def __init__(self, customers_data, source, membership_level):
        self.membership_level = membership_level
        super().__init__(customers_data, source)     
        self.customers_data = self.customers_data


    def filter_by_membership(self, level):
        return [
            customer
            for customer in self.customers_data
            if customer.get("membership") == level
        ]
    
    def display_info(self):
        super().display_info()
        print(f"Уровень членства: {self.membership_level}")



# customers = [
#     {"name": "Misha", "age": 33, "membership": "silver"},
#     {"name": "Anna", "age": 28, "membership": "gold"},
#     {"name": "Artem", "age": 12, "membership": "basic"},
#     {"name": "Mira", "age": 11 , "membership": "basic"},
#     {"name": "Max", "age": 43 , "membership": "gold"},
#     {'id': 1, 'name': 'John', 'age': 30, 'membership': 'gold'}
# ]

# gold_dataset = PremiumCustomerDataset(customers, "API", 'gold')
# gold_dataset.display_info() 
# print(gold_dataset.customers_data)

# print("""Фильтр по уровню членства:""")

# print(gold_dataset.filter_by_membership("basic"))