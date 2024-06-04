class relation:
    def __init__(self, value, classes=['empty', 'nok', 'ok'], answer=[True, True, True]):
        for prop, result in zip(classes, answer):
            self.__setattr__(prop, result if  classes.index(prop)==value else not(result))
    
    def __str__(self) -> str:
        return str(vars(self))
    
    def __repr__(self) -> str:
        return self.__str__()

def create_relation(result, f2d, f2v, roi):
    return frame(f2d, f2v, roi, **vars(relation(result, classes=['presence', 'orientation'], answer=[False, False])))
