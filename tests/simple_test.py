#!/usr/bin/env python3
"""
Teste simples de funcionalidade
"""

import sys
print("Python version:", sys.version)

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__} importado com sucesso")
except Exception as e:
    print(f"❌ Erro ao importar pandas: {e}")
    exit(1)

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} importado com sucesso")
except Exception as e:
    print(f"❌ Erro ao importar numpy: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__} importado com sucesso")
except Exception as e:
    print(f"❌ Erro ao importar sklearn: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib importado com sucesso")
except Exception as e:
    print(f"❌ Erro ao importar matplotlib: {e}")

# Teste básico de funcionalidade
try:
    # Criar dados simples de teste
    data = {
        'semana': [1, 2, 3, 4, 5],
        'pdv': [1001, 1001, 1002, 1002, 1003],
        'sku': ['A001', 'A002', 'A001', 'A003', 'A001'],
        'quantidade': [10, 15, 8, 20, 12]
    }
    
    df = pd.DataFrame(data)
    print("\n✅ DataFrame de teste criado:")
    print(df)
    
    # Operações básicas
    print(f"\n📊 Estatísticas básicas:")
    print(f"Total de registros: {len(df)}")
    print(f"Quantidade total: {df['quantidade'].sum()}")
    print(f"Média por registro: {df['quantidade'].mean():.2f}")
    
    print("\n🎉 Todas as funcionalidades básicas funcionam!")
    
except Exception as e:
    print(f"❌ Erro no teste de funcionalidade: {e}")
    import traceback
    traceback.print_exc()
