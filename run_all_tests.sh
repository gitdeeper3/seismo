#!/bin/bash
echo "🚀 بدء تشغيل جميع اختبارات Seismo Framework"
echo "=========================================="
echo ""

tests=(
    "test_seismo.py"
    "test_basic_functionality.py"
    "test_complete_model.py"
    "test_no_scipy.py"
    "minimal_test.py"
    "practical_example.py"
    "enhanced_test.py"
    "final_validation.py"
    "organize_project.py"
)

total=0
passed=0

for test in "${tests[@]}"; do
    if [ -f "tests/$test" ]; then
        ((total++))
        echo "🧪 تشغيل: $test"
        echo "------------------------------------------"
        
        if python "tests/$test" 2>&1; then
            echo "✅ $test - نجح"
            ((passed++))
        else
            echo "❌ $test - فشل"
        fi
        
        echo ""
        sleep 1
    fi
done

echo "=========================================="
echo "النتيجة النهائية: $passed/$total اختبارات ناجحة"
echo "نسبة النجاح: $((passed * 100 / total))%"
