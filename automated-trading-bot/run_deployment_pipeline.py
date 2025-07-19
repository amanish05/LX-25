#!/usr/bin/env python3
"""
Automated Trading Bot - Deployment Pipeline
Consolidated workflow: optimize → validate → test → train → report
Run this after every significant change before deployment
"""

import sys
import subprocess
import time
from datetime import datetime
import json

class DeploymentPipeline:
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {
            'pipeline_start': self.start_time.isoformat(),
            'steps': {}
        }
    
    def log(self, message, status="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
        print(f"[{timestamp}] {status_emoji.get(status, 'ℹ️')} {message}")
    
    def run_command(self, command, step_name, timeout=300):
        """Run a command and track results"""
        self.log(f"Starting: {step_name}")
        start_time = time.time()
        
        try:
            # Activate virtual environment and run command
            if isinstance(command, str):
                full_command = f"source venv/bin/activate && {command}"
            else:
                full_command = "source venv/bin/activate && " + " ".join(command)
            
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.log(f"Completed: {step_name} ({duration:.1f}s)", "SUCCESS")
                self.results['steps'][step_name] = {
                    'status': 'success',
                    'duration': duration,
                    'output': result.stdout[-500:] if result.stdout else ""  # Last 500 chars
                }
                return True
            else:
                self.log(f"Failed: {step_name} - {result.stderr[:200]}", "ERROR")
                self.results['steps'][step_name] = {
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr[:500] if result.stderr else ""
                }
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"Timeout: {step_name} (>{timeout}s)", "ERROR")
            self.results['steps'][step_name] = {
                'status': 'timeout',
                'duration': timeout,
                'error': f"Command timed out after {timeout}s"
            }
            return False
        except Exception as e:
            self.log(f"Error: {step_name} - {str(e)}", "ERROR")
            self.results['steps'][step_name] = {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
            return False
    
    def step_1_optimize_system(self):
        """Step 1: Run system optimization (if needed)"""
        self.log("STEP 1: System Optimization", "INFO")
        
        # Check if we have current optimization
        try:
            with open('config/price_action_fine_tuned.json', 'r') as f:
                config = json.load(f)
            
            # Check if optimization is recent (within 7 days)
            if 'optimization_metadata' in config:
                opt_date = config['optimization_metadata'].get('fine_tuned_date')
                if opt_date:
                    opt_time = datetime.fromisoformat(opt_date.replace('Z', '+00:00'))
                    days_old = (datetime.now() - opt_time.replace(tzinfo=None)).days
                    
                    if days_old < 7:
                        self.log("Optimization is recent, skipping", "SUCCESS")
                        self.results['steps']['optimization'] = {
                            'status': 'skipped',
                            'reason': f'Recent optimization ({days_old} days old)'
                        }
                        return True
            
            self.log("Running parameter optimization", "WARNING")
            return self.run_command("python src/optimization/run_optimization.py", "optimization", timeout=600)
            
        except FileNotFoundError:
            self.log("No optimization config found, running optimization", "WARNING")
            return self.run_command("python src/optimization/run_optimization.py", "optimization", timeout=600)
    
    def step_2_validate_system(self):
        """Step 2: Validate system configuration and components"""
        self.log("STEP 2: System Validation", "INFO")
        return self.run_command("python tests/validation/validate_system.py", "validation")
    
    def step_3_run_tests(self):
        """Step 3: Run comprehensive test suite"""
        self.log("STEP 3: Test Suite", "INFO")
        return self.run_command("DEPLOYMENT_PIPELINE=1 ./tests/scripts/run_tests.sh fast", "tests", timeout=180)
    
    def step_4_train_model(self):
        """Step 4: Train/update ML models"""
        self.log("STEP 4: Model Training", "INFO")
        
        # Check if we should skip training (if recent model exists)
        try:
            import os
            import json
            from datetime import datetime
            
            if os.path.exists('reports/model_training_report.json'):
                with open('reports/model_training_report.json', 'r') as f:
                    report = json.load(f)
                
                if 'training_date' in report:
                    train_date = datetime.fromisoformat(report['training_date'])
                    days_old = (datetime.now() - train_date).days
                    
                    if days_old < 7:
                        self.log(f"Model is recent ({days_old} days old), skipping training", "SUCCESS")
                        self.results['steps']['model_training'] = {
                            'status': 'skipped',
                            'reason': f'Recent model ({days_old} days old)'
                        }
                        return True
            
            self.log("Running model training pipeline", "WARNING")
            return self.run_command("python src/optimization/model_training_pipeline.py", "model_training", timeout=600)
            
        except Exception as e:
            self.log(f"Model training check failed: {e}", "WARNING")
            return self.run_command("python src/optimization/model_training_pipeline.py", "model_training", timeout=600)
    
    def step_5_generate_reports(self):
        """Step 5: Generate performance reports and visualizations"""
        self.log("STEP 5: Performance Reports", "INFO")
        
        # Generate performance visualizations
        success = self.run_command("python reports/visualize_performance.py", "visualizations")
        
        if success:
            # Generate ML performance comparison
            ml_success = self.run_command("python reports/visualize_trained_vs_actual.py", "ml_visualizations")
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Clean up reports (keep only best/last 2 versions)
            report_cleanup = self.run_command("python reports/report_versioning.py", "report_cleanup")
            
            return success and ml_success and report_cleanup
        
        return success
    
    def update_performance_metrics(self):
        """Update performance metrics with current timestamp"""
        try:
            with open('reports/performance_metrics.json', 'r') as f:
                metrics = json.load(f)
            
            metrics['last_pipeline_run'] = datetime.now().isoformat()
            metrics['pipeline_version'] = 'deployment_pipeline_v1'
            
            with open('reports/performance_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.log("Performance metrics updated", "SUCCESS")
            
        except Exception as e:
            self.log(f"Failed to update performance metrics: {e}", "WARNING")
    
    def generate_pipeline_report(self):
        """Generate pipeline execution report"""
        self.results['pipeline_end'] = datetime.now().isoformat()
        self.results['total_duration'] = (datetime.now() - self.start_time).total_seconds()
        
        # Count successes/failures
        steps = self.results['steps']
        total_steps = len(steps)
        successful_steps = sum(1 for step in steps.values() if step.get('status') == 'success')
        failed_steps = sum(1 for step in steps.values() if step.get('status') == 'failed')
        
        self.results['summary'] = {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': f"{successful_steps/total_steps*100:.1f}%" if total_steps > 0 else "0%"
        }
        
        # Save detailed report
        with open('reports/pipeline_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("DEPLOYMENT PIPELINE SUMMARY")
        print("="*60)
        print(f"Duration: {self.results['total_duration']:.1f} seconds")
        print(f"Steps: {successful_steps}/{total_steps} successful")
        
        if failed_steps == 0:
            print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
            print("✅ System is ready for deployment")
            return True
        else:
            print(f"❌ {failed_steps} steps failed")
            print("⚠️  Review issues before deployment")
            
            # Show failed steps
            print("\nFailed steps:")
            for step_name, step_result in steps.items():
                if step_result.get('status') == 'failed':
                    print(f"  - {step_name}: {step_result.get('error', 'Unknown error')[:100]}")
            
            return False
    
    def run_full_pipeline(self):
        """Run the complete deployment pipeline"""
        print("🚀 Starting Automated Trading Bot Deployment Pipeline")
        print("="*60)
        
        steps = [
            ("System Optimization", self.step_1_optimize_system),
            ("System Validation", self.step_2_validate_system),
            ("Test Suite", self.step_3_run_tests),
            ("Model Training", self.step_4_train_model),
            ("Performance Reports", self.step_5_generate_reports)
        ]
        
        all_success = True
        
        for step_name, step_func in steps:
            try:
                success = step_func()
                if not success:
                    all_success = False
                    self.log(f"Pipeline step failed: {step_name}", "ERROR")
                    
                    # Ask if user wants to continue
                    response = input(f"\nStep '{step_name}' failed. Continue with next steps? (y/N): ")
                    if response.lower() != 'y':
                        self.log("Pipeline aborted by user", "WARNING")
                        break
                        
            except KeyboardInterrupt:
                self.log("Pipeline interrupted by user", "WARNING")
                break
            except Exception as e:
                self.log(f"Unexpected error in {step_name}: {e}", "ERROR")
                all_success = False
                break
        
        # Generate final report
        pipeline_success = self.generate_pipeline_report()
        
        return pipeline_success and all_success


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Automated Trading Bot - Deployment Pipeline")
            print("Usage: python run_deployment_pipeline.py [step]")
            print("")
            print("Steps:")
            print("  optimize   - Run system optimization only")
            print("  validate   - Run system validation only")
            print("  test       - Run test suite only")
            print("  train      - Run model training only")
            print("  report     - Generate reports only")
            print("  full       - Run complete pipeline (default)")
            return
        
        # Run individual step
        pipeline = DeploymentPipeline()
        step = sys.argv[1].lower()
        
        if step == 'optimize':
            success = pipeline.step_1_optimize_system()
        elif step == 'validate':
            success = pipeline.step_2_validate_system()
        elif step == 'test':
            success = pipeline.step_3_run_tests()
        elif step == 'train':
            success = pipeline.step_4_train_model()
        elif step == 'report':
            success = pipeline.step_5_generate_reports()
        else:
            print(f"Unknown step: {step}")
            return
        
        if success:
            print(f"✅ Step '{step}' completed successfully")
        else:
            print(f"❌ Step '{step}' failed")
            sys.exit(1)
    
    else:
        # Run full pipeline
        pipeline = DeploymentPipeline()
        success = pipeline.run_full_pipeline()
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()