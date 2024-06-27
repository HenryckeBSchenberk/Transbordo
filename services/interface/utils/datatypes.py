class frame:
    def __init__(self, original_data, normalized_data, roi, **kwargs) -> None:
        self.original_data = original_data
        self.normalized_data = normalized_data
        self.roi = roi

        for k,v in kwargs.items():
            setattr(self, k, v)
        
    def update(self, attr, value):
        setattr(self, attr, value)
        return self
    
    def update_values(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        return self
    
    # def __str__(self) -> str:
    #     return (f"roi:{self.roi}, presence:{getattr(self, 'presence', 'Undefined'), 'orientation':getattr(self, 'orientation', 'Undefined')}")

    # def __repr__(self) -> str:
    #     return str(self)