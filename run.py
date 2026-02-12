"""
Run All Experiments Script V2.0
Với error handling, recovery, và lưu kết quả riêng biệt cho mỗi experiment
"""
import subprocess
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback


class ExperimentRunner:
    """Chạy experiments với error handling và recovery"""
    
    def __init__(self, base_dir='results'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"full_pipeline_{timestamp}"
        self.run_dir = self.base_dir / 'full_runs' / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Track results
        self.results = {
            'run_name': self.run_name,
            'started_at': datetime.now().isoformat(),
            'experiments': []
        }
        
        self.logger.info(f"Pipeline run initialized: {self.run_name}")
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.run_dir / 'pipeline.log'
        
        self.logger = logging.getLogger('ExperimentRunner')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def print_header(self, text):
        """Print formatted header"""
        self.logger.info("\n" + "="*80)
        self.logger.info(text)
        self.logger.info("="*80 + "\n")
    
    def run_command(self, cmd, experiment_name, description):
        """
        Chạy command với error handling
        
        Returns:
            (success: bool, error_msg: str or None)
        """
        self.logger.info(f"Running: {description}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        # Record start
        exp_result = {
            'name': experiment_name,
            'description': description,
            'command': ' '.join(cmd),
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        try:
            # Run command
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Success
            exp_result['status'] = 'completed'
            exp_result['completed_at'] = datetime.now().isoformat()
            exp_result['stdout'] = result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout  # Last 1000 chars
            
            self.logger.info(f"✓ {description} completed successfully\n")
            self.results['experiments'].append(exp_result)
            self.save_results()
            
            return True, None
        
        except subprocess.CalledProcessError as e:
            # Failure
            error_msg = f"Command failed with return code {e.returncode}"
            exp_result['status'] = 'failed'
            exp_result['failed_at'] = datetime.now().isoformat()
            exp_result['error'] = error_msg
            exp_result['stderr'] = e.stderr[-1000:] if e.stderr and len(e.stderr) > 1000 else e.stderr
            
            self.logger.error(f"✗ {description} failed: {error_msg}")
            if e.stderr:
                self.logger.error(f"Error output:\n{e.stderr[-500:]}")  # Last 500 chars
            
            self.results['experiments'].append(exp_result)
            self.save_results()
            
            return False, error_msg
        
        except KeyboardInterrupt:
            exp_result['status'] = 'interrupted'
            exp_result['interrupted_at'] = datetime.now().isoformat()
            self.results['experiments'].append(exp_result)
            self.save_results()
            
            self.logger.warning(f"⚠ {description} interrupted by user")
            raise
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            exp_result['status'] = 'error'
            exp_result['error'] = error_msg
            exp_result['traceback'] = traceback.format_exc()
            
            self.logger.error(f"✗ {description} error: {error_msg}")
            self.logger.error(traceback.format_exc())
            
            self.results['experiments'].append(exp_result)
            self.save_results()
            
            return False, error_msg
    
    def save_results(self):
        """Save results to JSON"""
        results_file = self.run_dir / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def run_all(self, experiments, stop_on_error=False, skip_failed=False):
        """
        Chạy tất cả experiments
        
        Args:
            experiments: list of experiment configs
            stop_on_error: dừng nếu có lỗi
            skip_failed: bỏ qua experiments đã failed trước đó
        """
        self.print_header(f"FULL PIPELINE RUN: {self.run_name}")
        
        # Check data
        if not self.check_data():
            self.logger.error("Data not found! Aborting.")
            return False
        
        # Run each experiment
        failed_experiments = []
        
        for i, exp in enumerate(experiments, 1):
            if i == 1:
                continue
            self.print_header(f"STEP {i}/{len(experiments)}: {exp['description']}")
            
            success, error = self.run_command(
                exp['cmd'],
                exp['name'],
                exp['description']
            )
            
            if not success:
                failed_experiments.append(exp['name'])
                
                if stop_on_error:
                    self.logger.error(f"Stopping pipeline due to error in {exp['name']}")
                    break
                else:
                    self.logger.warning(f"Continuing despite error in {exp['name']}")
        
        # Final summary
        self.results['completed_at'] = datetime.now().isoformat()
        self.results['total_experiments'] = len(experiments)
        self.results['successful'] = len([e for e in self.results['experiments'] if e['status'] == 'completed'])
        self.results['failed'] = len(failed_experiments)
        self.results['failed_experiments'] = failed_experiments
        self.save_results()
        
        self.print_summary()
        
        return len(failed_experiments) == 0
    
    def check_data(self):
        """Check if data exists"""
        data_path = Path("data/hyper-kvasir/labeled-images")
        if not data_path.exists():
            return False
        self.logger.info("✓ Dataset found\n")
        return True
    
    def print_summary(self):
        """Print final summary"""
        self.print_header("PIPELINE EXECUTION SUMMARY")
        
        total = self.results['total_experiments']
        successful = self.results['successful']
        failed = self.results['failed']
        
        self.logger.info(f"Total experiments: {total}")
        self.logger.info(f"Successful: {successful} ({successful/total*100:.1f}%)")
        self.logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        if self.results['failed_experiments']:
            self.logger.info("\nFailed experiments:")
            for exp_name in self.results['failed_experiments']:
                self.logger.info(f"  ✗ {exp_name}")
        
        self.logger.info(f"\nResults saved to: {self.run_dir}")
        self.logger.info(f"  - Pipeline results: {self.run_dir / 'pipeline_results.json'}")
        self.logger.info(f"  - Log file: {self.run_dir / 'pipeline.log'}")
        
        if failed == 0:
            self.logger.info("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        else:
            self.logger.warning(f"\n⚠️  {failed} EXPERIMENT(S) FAILED")
        
        self.logger.info("\nTo view individual experiment results:")
        self.logger.info("  Training: results/training/")
        self.logger.info("  Benchmark: results/benchmark_comparison/")
        self.logger.info("  Ablation: results/ablation_study/")


def main():
    """Main function"""
    # Create runner
    runner = ExperimentRunner()
    
    # Define experiments với tên unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiments = [
        {
            'name': f'training_{timestamp}',
            'cmd': [
                sys.executable,
                'experiments/train.py',
                '--batch_size', '8',
                '--epochs', '100',
                '--lr', '0.0001',
                '--mixup_alpha', '0.4',
                '--use_mixup',
                '--use_unlabeled',
                '--experiment_name', f'training_{timestamp}'  # Pass unique name
            ],
            'description': 'Training ResNet50-UNet with Mixup'
        },
        {
            'name': f'benchmark_{timestamp}',
            'cmd': [
                sys.executable,
                'experiments/benchmark_comparison.py',
                '--experiment_name', f'benchmark_{timestamp}'
            ],
            'description': 'Benchmark Comparison'
        },
        {
            'name': f'ablation_{timestamp}',
            'cmd': [
                sys.executable,
                'experiments/ablation_study.py',
                '--experiment_name', f'ablation_{timestamp}'
            ],
            'description': 'Ablation Study'
        }
    ]
    
    success = runner.run_all(
        experiments,
        stop_on_error=False 
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)