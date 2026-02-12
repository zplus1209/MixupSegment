"""
Experiment Manager - Quản lý experiments với error handling và recovery
"""
import os
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
import traceback


class ExperimentManager:
    """
    Quản lý experiments với:
    - Tự động tạo tên unique cho mỗi lần chạy
    - Lưu config và results riêng biệt
    - Error handling và recovery
    - Logging đầy đủ
    """
    
    def __init__(self, experiment_type, base_dir='results', experiment_name=None):
        """
        Args:
            experiment_type: 'training', 'benchmark', 'ablation', 'inference'
            base_dir: thư mục gốc lưu results
            experiment_name: tên custom (nếu None sẽ tự generate)
        """
        self.experiment_type = experiment_type
        self.base_dir = Path(base_dir)
        
        # Generate unique experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{experiment_type}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Create experiment directory structure
        self.exp_dir = self.base_dir / experiment_type / self.experiment_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.vis_dir = self.exp_dir / 'visualizations'
        self.config_dir = self.exp_dir / 'configs'
        
        # Create all directories
        for d in [self.checkpoint_dir, self.log_dir, self.vis_dir, self.config_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Track experiment state
        self.state_file = self.exp_dir / 'experiment_state.json'
        self.state = self.load_state()
        
        self.logger.info(f"Experiment Manager initialized: {self.experiment_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")
    
    def setup_logging(self):
        """Setup logging cho experiment"""
        log_file = self.log_dir / f'{self.experiment_name}.log'
        
        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def save_config(self, config, name='config.json'):
        """Lưu config của experiment"""
        config_file = self.config_dir / name
        
        # Convert non-serializable objects
        config_serializable = {}
        for key, value in config.items():
            if isinstance(value, (Path, )):
                config_serializable[key] = str(value)
            else:
                config_serializable[key] = value
        
        with open(config_file, 'w') as f:
            json.dump(config_serializable, f, indent=4)
        
        self.logger.info(f"Config saved to {config_file}")
    
    def load_state(self):
        """Load experiment state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.logger.info(f"Loaded existing experiment state")
            return state
        else:
            return {
                'status': 'initialized',
                'created_at': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'completed': False,
                'error': None,
                'checkpoints': [],
                'results': {}
            }
    
    def save_state(self):
        """Save experiment state"""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)
    
    def update_status(self, status, error=None):
        """Update experiment status"""
        self.state['status'] = status
        if error:
            self.state['error'] = str(error)
            self.logger.error(f"Error: {error}")
        self.save_state()
    
    def log_checkpoint(self, checkpoint_path, epoch, metrics):
        """Log checkpoint information"""
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.state['checkpoints'].append(checkpoint_info)
        self.save_state()
        self.logger.info(f"Checkpoint saved: epoch {epoch}, metrics: {metrics}")
    
    def save_results(self, results, name='results.json'):
        """Save final results"""
        results_file = self.exp_dir / name
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.state['results'] = results
        self.state['completed'] = True
        self.save_state()
        self.logger.info(f"Results saved to {results_file}")
    
    def get_checkpoint_path(self, name='checkpoint.pth'):
        """Get path for checkpoint"""
        return self.checkpoint_dir / name
    
    def get_visualization_path(self, name):
        """Get path for visualization"""
        return self.vis_dir / name
    
    def find_latest_checkpoint(self):
        """Find latest checkpoint in experiment"""
        if not self.state['checkpoints']:
            return None
        
        latest = max(self.state['checkpoints'], key=lambda x: x['epoch'])
        checkpoint_path = Path(latest['path'])
        
        if checkpoint_path.exists():
            self.logger.info(f"Found latest checkpoint: {checkpoint_path}")
            return checkpoint_path
        else:
            self.logger.warning(f"Latest checkpoint not found: {checkpoint_path}")
            return None
    
    def backup_checkpoint(self, checkpoint_path):
        """Backup checkpoint"""
        if not Path(checkpoint_path).exists():
            return
        
        backup_dir = self.checkpoint_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}_{Path(checkpoint_path).name}"
        
        shutil.copy2(checkpoint_path, backup_path)
        self.logger.info(f"Checkpoint backed up to {backup_path}")
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Keep only last N checkpoints"""
        if len(self.state['checkpoints']) <= keep_last_n:
            return
        
        # Sort by epoch
        sorted_checkpoints = sorted(self.state['checkpoints'], key=lambda x: x['epoch'])
        
        # Remove old checkpoints
        to_remove = sorted_checkpoints[:-keep_last_n]
        for checkpoint_info in to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists() and 'best' not in checkpoint_path.name:
                checkpoint_path.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update state
        self.state['checkpoints'] = sorted_checkpoints[-keep_last_n:]
        self.save_state()
    
    def handle_error(self, error, context=""):
        """Handle and log error"""
        error_msg = f"{context}: {str(error)}"
        self.logger.error(error_msg)
        self.logger.error(traceback.format_exc())
        
        self.update_status('failed', error_msg)
        
        # Save error report
        error_report = {
            'error': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        error_file = self.exp_dir / 'error_report.json'
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=4)
        
        self.logger.info(f"Error report saved to {error_file}")


def run_with_error_handling(func, experiment_manager, *args, **kwargs):
    """
    Wrapper để chạy function với error handling
    
    Usage:
        em = ExperimentManager('training')
        run_with_error_handling(train_model, em, model, dataloader)
    """
    try:
        experiment_manager.update_status('running')
        result = func(*args, **kwargs)
        experiment_manager.update_status('completed')
        return result
    
    except KeyboardInterrupt:
        experiment_manager.logger.warning("Experiment interrupted by user")
        experiment_manager.update_status('interrupted')
        raise
    
    except Exception as e:
        experiment_manager.handle_error(e, context=f"Running {func.__name__}")
        raise


class ExperimentTracker:
    """Track multiple experiments and compare results"""
    
    def __init__(self, base_dir='results'):
        self.base_dir = Path(base_dir)
        self.experiments = []
        self.load_experiments()
    
    def load_experiments(self):
        """Load all experiments from base directory"""
        for exp_type_dir in self.base_dir.iterdir():
            if exp_type_dir.is_dir():
                for exp_dir in exp_type_dir.iterdir():
                    if exp_dir.is_dir():
                        state_file = exp_dir / 'experiment_state.json'
                        if state_file.exists():
                            with open(state_file, 'r') as f:
                                state = json.load(f)
                            state['name'] = exp_dir.name
                            state['type'] = exp_type_dir.name
                            state['path'] = str(exp_dir)
                            self.experiments.append(state)
    
    def get_experiments_by_type(self, exp_type):
        """Get all experiments of a specific type"""
        return [e for e in self.experiments if e['type'] == exp_type]
    
    def get_completed_experiments(self):
        """Get all completed experiments"""
        return [e for e in self.experiments if e.get('completed', False)]
    
    def get_failed_experiments(self):
        """Get all failed experiments"""
        return [e for e in self.experiments if e.get('status') == 'failed']
    
    def print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print(f"\nTotal experiments: {len(self.experiments)}")
        print(f"Completed: {len(self.get_completed_experiments())}")
        print(f"Failed: {len(self.get_failed_experiments())}")
        
        print("\nExperiments by type:")
        for exp_type in set(e['type'] for e in self.experiments):
            exps = self.get_experiments_by_type(exp_type)
            print(f"  {exp_type}: {len(exps)}")
        
        print("\nRecent experiments:")
        recent = sorted(self.experiments, key=lambda x: x.get('last_update', ''), reverse=True)[:5]
        for exp in recent:
            print(f"  {exp['name']} ({exp['type']}): {exp['status']}")
        print()