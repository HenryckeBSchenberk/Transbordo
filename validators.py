steps = { 
    'presence': {
        'validator': './model/models/presence/best',
        'threshold': lambda r: r[-1] > 0.5,
    },
    'orientation':{
        'validator':'./model/models/orientation/best',
        'threshold':lambda r: r[-1] > 0.5
    },
}