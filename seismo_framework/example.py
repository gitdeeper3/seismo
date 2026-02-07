"""
Example usage of Seismo Framework.
"""
import numpy as np
from seismo_framework import (
    SeismicAnalyzer,
    ParameterIntegrator,
    RealTimeMonitor,
    about,
    quick_start
)
def test_parameters():
    """Test parameter analyzers."""
    print("Testing Seismic Analyzer...")
    # Create seismic analyzer
    seismic = SeismicAnalyzer()
    # Generate sample seismic data
    np.random.seed(42)
    sample_data = {
        'waveforms': np.random.randn(1000, 3),
        'sampling_rate': 100,
        'station_locations': [(40.5, 15.5, 1000)]
    }
    # Analyze
    result = seismic.analyze(sample_data)
    print(f"Seismic Index: {result.get('seismic_index', 0):.3f}")
    print(f"Event Count: {result.get('event_count', 0)}")
    return result
def test_integration():
    """Test parameter integration."""
    print("\nTesting Parameter Integration...")
    integrator = ParameterIntegrator()
    # Sample parameter values
    parameters = {
        'seismic': 0.75,
        'deformation': 0.60,
        'hydrogeological': 0.45,
        'electrical': 0.80,
        'magnetic': 0.55,
        'instability': 0.65,
        'stress': 0.70,
        'rock_properties': 0.50
    }
    result = integrator.integrate(parameters)
    print(f"Integrated Score: {result.get('integrated_score', 0):.3f}")
    print(f"Alert Level: {result.get('alert_level', 'normal')}")
    return result
def test_monitoring():
    """Test real-time monitoring."""
    print("\nTesting Real-time Monitoring...")
    # Create monitor
    monitor = RealTimeMonitor(config={
        'monitoring_interval': 5  # 5 seconds for testing
    })
    # Start monitoring
    monitor.start_monitoring()
    # Let it run for a bit
    import time
    print("Monitoring for 15 seconds...")
    time.sleep(15)
    # Get current state
    state = monitor.get_current_state()
    print(f"Current Alert: {state.get('current_alert', {})}")
    # Stop monitoring
    monitor.stop_monitoring()
    return monitor
def main():
    """Run all tests."""
    print("=" * 60)
    print("SEISMO FRAMEWORK - TEST SUITE")
    print("=" * 60)
    about()
    try:
        # Test 1: Parameter analysis
        seismic_result = test_parameters()
        # Test 2: Integration
        integration_result = test_integration()
        # Test 3: Monitoring
        monitor = test_monitoring()
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check the generated plots and data")
        print("2. Review the alert levels")
        print("3. Consult the documentation for advanced usage")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()