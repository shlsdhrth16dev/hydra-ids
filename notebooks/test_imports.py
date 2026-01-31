"""Quick test of imports"""
print("Starting imports...")

try:
    print("1. Importing HealthMonitor...")
    from self_healing import HealthMonitor
    print("   OK")
    
    print("2. Importing DriftDetector...")
    from self_healing import DriftDetector
    print("   OK")
    
    print("3. Importing DecisionEngine...")
    from self_healing import DecisionEngine
    print("   OK")
    
    print("4. Importing Retrainer...")
    from self_healing import Retrainer
    print("   OK")
    
    print("5. Importing RollbackManager...")
    from self_healing import RollbackManager
    print("   OK")
    
    print("6. Importing AlertSystem...")
    from self_healing import AlertSystem
    print("   OK")
    
    print("7. Importing SelfHealingOrchestrator...")
    from self_healing import SelfHealingOrchestrator
    print("   OK")
    
    print("8. Importing SelfHealingConfig...")
    from self_healing import SelfHealingConfig
    print("   OK")
    
    print()
    print("ALL IMPORTS SUCCESSFUL!")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
