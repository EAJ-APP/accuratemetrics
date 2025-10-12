"""
Script de diagnóstico para CausalImpact
Ejecutar: python test_causal_impact.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("DIAGNÓSTICO DE CAUSALIMPACT")
print("=" * 80)

# 1. Verificar instalación de pycausalimpact
print("\n1️⃣ Verificando instalación de pycausalimpact...")
try:
    from causalimpact import CausalImpact
    print("✅ pycausalimpact está instalado")
    
    # Intentar obtener versión
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("pycausalimpact").version
        print(f"   Versión: {version}")
    except:
        print("   Versión: No se pudo determinar")
except ImportError:
    print("❌ pycausalimpact NO está instalado")
    print("   Instala con: pip install pycausalimpact")
    exit(1)

# 2. Crear datos de prueba
print("\n2️⃣ Creando datos de prueba...")
np.random.seed(42)

# Generar 90 días de datos
dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
baseline = 1000

# Pre-intervención: datos normales con algo de ruido
pre_data = baseline + np.random.normal(0, 100, 60)

# Post-intervención: aumento del 20%
post_data = baseline * 1.2 + np.random.normal(0, 100, 30)

# Combinar
sessions = np.concatenate([pre_data, post_data])

# Crear DataFrame
df = pd.DataFrame({
    'date': dates,
    'sessions': sessions
})

print(f"✅ Creados {len(df)} días de datos")
print(f"   Rango: {df['date'].min().date()} a {df['date'].max().date()}")
print(f"   Media pre (días 1-60): {sessions[:60].mean():.2f}")
print(f"   Media post (días 61-90): {sessions[60:].mean():.2f}")
print(f"   Cambio real: +20%")

# 3. Preparar datos
print("\n3️⃣ Preparando datos para CausalImpact...")
df_analysis = df.set_index('date')[['sessions']]
df_analysis.index = pd.DatetimeIndex(df_analysis.index, freq='D')

print(f"✅ DataFrame preparado")
print(f"   Shape: {df_analysis.shape}")
print(f"   Index type: {type(df_analysis.index)}")
print(f"   Freq: {df_analysis.index.freq}")

# 4. Definir períodos
intervention_date = pd.Timestamp('2024-03-01')
pre_start = df_analysis.index.min()
pre_end = intervention_date - pd.Timedelta(days=1)
post_start = intervention_date
post_end = df_analysis.index.max()

print(f"\n4️⃣ Períodos definidos:")
print(f"   Pre:  {pre_start.date()} a {pre_end.date()} ({(pre_end - pre_start).days + 1} días)")
print(f"   Post: {post_start.date()} a {post_end.date()} ({(post_end - post_start).days + 1} días)")

pre_period = [pre_start, pre_end]
post_period = [post_start, post_end]

# 5. Ejecutar CausalImpact
print("\n5️⃣ Ejecutando CausalImpact...")
try:
    ci = CausalImpact(
        df_analysis,
        pre_period,
        post_period,
        model_args={'nseasons': 7}
    )
    print("✅ CausalImpact ejecutado exitosamente")
except TypeError as e:
    print(f"⚠️ TypeError con nseasons: {e}")
    print("   Intentando sin model_args...")
    try:
        ci = CausalImpact(
            df_analysis,
            pre_period,
            post_period
        )
        print("✅ CausalImpact ejecutado exitosamente (sin nseasons)")
    except Exception as e2:
        print(f"❌ Error ejecutando CausalImpact: {e2}")
        exit(1)
except Exception as e:
    print(f"❌ Error ejecutando CausalImpact: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Examinar resultado
print("\n6️⃣ Examinando resultado de CausalImpact...")

# Ver qué atributos tiene
print("\nAtributos disponibles:")
attrs = [attr for attr in dir(ci) if not attr.startswith('_')]
for attr in attrs[:20]:  # Mostrar primeros 20
    print(f"   - {attr}")

# 7. Intentar obtener summary
print("\n7️⃣ Intentando obtener summary...")

if hasattr(ci, 'summary_data'):
    print("✅ Encontrado: ci.summary_data")
    summary = ci.summary_data
    print(f"   Type: {type(summary)}")
    print(f"   Shape: {summary.shape if hasattr(summary, 'shape') else 'N/A'}")
    
    if isinstance(summary, pd.DataFrame):
        print(f"   Index: {summary.index.tolist()}")
        print(f"   Columns: {summary.columns.tolist()}")
        print("\n   Contenido:")
        print(summary)
elif hasattr(ci, 'summary'):
    print("✅ Encontrado: ci.summary()")
    try:
        summary = ci.summary()
        print(f"   Type: {type(summary)}")
        if isinstance(summary, pd.DataFrame):
            print(f"   Index: {summary.index.tolist()}")
            print(f"   Columns: {summary.columns.tolist()}")
            print("\n   Contenido:")
            print(summary)
    except Exception as e:
        print(f"❌ Error llamando summary(): {e}")
else:
    print("❌ No se encontró ni summary_data ni summary()")

# 8. Intentar obtener inferences
print("\n8️⃣ Examinando inferences...")

if hasattr(ci, 'inferences'):
    print("✅ Encontrado: ci.inferences")
    inferences = ci.inferences
    print(f"   Type: {type(inferences)}")
    print(f"   Shape: {inferences.shape}")
    print(f"   Columns: {inferences.columns.tolist()}")
    print(f"\n   Primeras 5 filas:")
    print(inferences.head())
    print(f"\n   Últimas 5 filas:")
    print(inferences.tail())
    
    # Calcular métricas manualmente
    print("\n   📊 Métricas calculadas manualmente:")
    post_mask = inferences.index >= intervention_date
    
    actual_post = inferences.loc[post_mask, 'response']
    pred_post = inferences.loc[post_mask, 'preds']
    
    actual_mean = actual_post.mean()
    pred_mean = pred_post.mean()
    effect_mean = actual_mean - pred_mean
    rel_effect = (effect_mean / pred_mean) * 100 if pred_mean != 0 else 0
    
    print(f"      Actual promedio post: {actual_mean:.2f}")
    print(f"      Predicho promedio post: {pred_mean:.2f}")
    print(f"      Efecto absoluto: {effect_mean:.2f}")
    print(f"      Efecto relativo: {rel_effect:.2f}%")
    
    actual_sum = actual_post.sum()
    pred_sum = pred_post.sum()
    effect_sum = actual_sum - pred_sum
    rel_effect_cum = (effect_sum / pred_sum) * 100 if pred_sum != 0 else 0
    
    print(f"\n      Actual acumulado: {actual_sum:.2f}")
    print(f"      Predicho acumulado: {pred_sum:.2f}")
    print(f"      Efecto acumulado: {effect_sum:.2f}")
    print(f"      Efecto relativo acumulado: {rel_effect_cum:.2f}%")
else:
    print("❌ No se encontró inferences")

# 9. P-value
print("\n9️⃣ P-value...")
if hasattr(ci, 'p_value'):
    print(f"✅ P-value: {ci.p_value}")
else:
    print("❌ No se encontró p_value")

# 10. Resumen final
print("\n" + "=" * 80)
print("RESUMEN DEL DIAGNÓSTICO")
print("=" * 80)

print("\n✅ ÉXITOS:")
print("   - pycausalimpact está instalado")
print("   - CausalImpact se ejecutó sin errores")
print("   - Los datos de prueba muestran un cambio del +20%")

print("\n📋 INFORMACIÓN CRÍTICA:")
if hasattr(ci, 'summary_data'):
    print("   - Usar ci.summary_data para obtener el resumen")
elif hasattr(ci, 'summary'):
    print("   - Usar ci.summary() para obtener el resumen")
else:
    print("   - No hay método directo para summary")
    print("   - Usar ci.inferences para calcular métricas manualmente")

print("\n💡 PRÓXIMOS PASOS:")
print("   1. Verifica que tu código use las columnas correctas:")
print("      - 'response' para valores reales")
print("      - 'preds' para valores predichos")
print("      - 'preds_lower' y 'preds_upper' para IC")
print("   2. Si summary está vacío, calcula desde inferences")
print("   3. Asegúrate de filtrar post_period correctamente")

print("\n" + "=" * 80)
print("Diagnóstico completado.")
print("=" * 80)
