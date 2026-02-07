"""
تشغيل محاكاة كاملة لنظام Seismo Framework
باستخدام البيانات والكود الحقيقي الموجود
"""

import sys
import os
sys.path.insert(0, '.')

from seismo_framework import SeismicAnalyzer, DeformationAnalyzer, ParameterIntegrator
from datetime import datetime

print("🌋 Seismo Framework - المحاكاة الكاملة")
print("=" * 60)

# 1. إنشاء المحللات
print("1. 🔧 تهيئة المحللات...")
seismic = SeismicAnalyzer()
deformation = DeformationAnalyzer()
integrator = ParameterIntegrator()

print("   ✅ تم إنشاء 3 محللين")

# 2. بيانات تجريبية
print("\n2. 📊 توليد بيانات تجريبية...")
seismic_data = {"magnitudes": [2.5, 3.0, 2.8, 3.5], "depths": [5, 10, 8, 12]}
deformation_data = {"displacements": [1.2, 0.8, 1.5, 0.9]}

print(f"   ✅ بيانات زلزالية: {len(seismic_data['magnitudes'])} زلزال")
print(f"   ✅ بيانات تشوه: {len(deformation_data['displacements'])} قياس")

# 3. التحليل
print("\n3. 🔍 تحليل البيانات...")
seismic_result = seismic.analyze(seismic_data)
deformation_result = deformation.analyze(deformation_data)

print(f"   📈 النتيجة الزلزالية: {seismic_result.get('seismic_index', 'N/A')}")
print(f"   📈 نتيجة التشوه: {deformation_result.get('deformation_index', 'N/A')}")

# 4. التكامل
print("\n4. 🔗 تكامل المعلمات...")
parameters = {
    'seismic': seismic_result.get('seismic_index', 0.5),
    'deformation': deformation_result.get('deformation_index', 0.5),
    'hydrogeological': 0.4,
    'electrical': 0.3,
    'magnetic': 0.6,
    'instability': 0.5,
    'stress': 0.4,
    'rock_properties': 0.3
}

integration_result = integrator.integrate(parameters)

print(f"   🎯 النتيجة المتكاملة: {integration_result.get('integrated_score', 0)}")
print(f"   🚨 مستوى الإنذار: {integration_result.get('alert_level', 'UNKNOWN')}")

# 5. الخلاصة
print("\n" + "=" * 60)
print("✅ محاكاة Seismo Framework اكتملت بنجاح!")
print(f"   الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
