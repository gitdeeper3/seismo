"""
Report manager for organizing and managing reports.
"""

import os
import shutil
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ReportManager:
    """
    Manages report organization and storage.
    """
    
    def __init__(self, base_dir: str = "reports"):
        self.base_dir = base_dir
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup report directory structure."""
        directories = [
            'daily',
            'weekly',
            'monthly',
            'alerts',
            'archived',
            'templates'
        ]
        
        for directory in directories:
            dir_path = os.path.join(self.base_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
    
    def organize_existing_reports(self, source_dir: str = "."):
        """
        Organize existing reports into proper structure.
        
        Args:
            source_dir: Directory containing existing reports
        """
        logger.info(f"Organizing reports from {source_dir}")
        
        report_files = self._find_report_files(source_dir)
        
        for file_path in report_files:
            try:
                self._organize_file(file_path)
                logger.info(f"Organized: {file_path}")
            except Exception as e:
                logger.error(f"Error organizing {file_path}: {e}")
    
    def _find_report_files(self, directory: str) -> List[str]:
        """Find report files in directory."""
        report_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if self._is_report_file(file):
                    file_path = os.path.join(root, file)
                    report_files.append(file_path)
        
        return report_files
    
    def _is_report_file(self, filename: str) -> bool:
        """Check if file is a report file."""
        report_patterns = [
            'report_', 'seismo_', 'analysis_', 'data_',
            '.txt', '.csv', '.json', '.pdf', '.html'
        ]
        
        filename_lower = filename.lower()
        return any(pattern in filename_lower for pattern in report_patterns)
    
    def _organize_file(self, file_path: str):
        """Organize a single file into proper directory."""
        filename = os.path.basename(file_path)
        
        # Determine file type and destination
        if 'alert' in filename.lower():
            dest_dir = 'alerts'
        elif 'daily' in filename.lower():
            dest_dir = 'daily'
        elif 'weekly' in filename.lower():
            dest_dir = 'weekly'
        elif 'monthly' in filename.lower():
            dest_dir = 'monthly'
        else:
            # Try to determine from date in filename
            dest_dir = self._determine_directory_from_date(filename)
        
        # Create destination path
        dest_path = os.path.join(self.base_dir, dest_dir, filename)
        
        # Move file (copy to avoid data loss)
        shutil.copy2(file_path, dest_path)
        logger.debug(f"Copied {file_path} to {dest_path}")
    
    def _determine_directory_from_date(self, filename: str) -> str:
        """Determine directory based on date in filename."""
        try:
            # Try to extract date from filename
            # Look for patterns like YYYYMMDD, YYYY-MM-DD, etc.
            import re
            
            date_patterns = [
                r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
                r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
                r'(\d{4})_(\d{2})_(\d{2})',  # YYYY_MM_DD
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, filename)
                if match:
                    year, month, day = match.groups()
                    file_date = datetime(int(year), int(month), int(day))
                    
                    # Determine if file is old enough to archive
                    days_old = (datetime.now() - file_date).days
                    
                    if days_old > 90:  # Archive old reports
                        return 'archived'
                    elif days_old > 30:  # Monthly
                        return 'monthly'
                    elif days_old > 7:   # Weekly
                        return 'weekly'
                    else:                # Daily
                        return 'daily'
        
        except Exception:
            pass
        
        # Default to daily for undetermined files
        return 'daily'
    
    def cleanup_old_reports(self, days_to_keep: int = 90):
        """
        Clean up reports older than specified days.
        
        Args:
            days_to_keep: Number of days to keep reports
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for report_type in ['daily', 'weekly', 'monthly', 'alerts']:
            dir_path = os.path.join(self.base_dir, report_type)
            
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_time < cutoff_date:
                            # Move to archived or delete
                            archive_path = os.path.join(self.base_dir, 'archived', filename)
                            shutil.move(file_path, archive_path)
                            logger.info(f"Archived old report: {filename}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of all reports."""
        summary = {
            'total_reports': 0,
            'by_type': {},
            'by_directory': {},
            'total_size_mb': 0
        }
        
        for root, dirs, files in os.walk(self.base_dir):
            dir_name = os.path.basename(root)
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Update counts
                summary['total_reports'] += 1
                
                # Update by directory
                if dir_name not in summary['by_directory']:
                    summary['by_directory'][dir_name] = 0
                summary['by_directory'][dir_name] += 1
                
                # Update by file type
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in summary['by_type']:
                    summary['by_type'][file_ext] = 0
                summary['by_type'][file_ext] += 1
                
                # Update total size
                try:
                    file_size = os.path.getsize(file_path)
                    summary['total_size_mb'] += file_size / (1024 * 1024)
                except:
                    pass
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        
        return summary
    
    def generate_index(self) -> str:
        """Generate HTML index of all reports."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Seismo Framework - Reports Index</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .alert { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🌋 Seismo Framework - Reports Index</h1>
            <p>Generated: {timestamp}</p>
            <hr>
        """.format(timestamp=datetime.now().isoformat())
        
        # Add summary
        summary = self.get_report_summary()
        html += f"""
        <h2>📊 Summary</h2>
        <p>Total Reports: {summary['total_reports']}</p>
        <p>Total Size: {summary['total_size_mb']} MB</p>
        """
        
        # Add directory listing
        for directory in ['daily', 'weekly', 'monthly', 'alerts', 'archived']:
            dir_path = os.path.join(self.base_dir, directory)
            
            if os.path.exists(dir_path):
                files = sorted(os.listdir(dir_path))
                
                if files:
                    html += f"""
                    <h2>📁 {directory.capitalize()} Reports</h2>
                    <table>
                        <tr>
                            <th>Filename</th>
                            <th>Size</th>
                            <th>Modified</th>
                        </tr>
                    """
                    
                    for file in files:
                        file_path = os.path.join(dir_path, file)
                        try:
                            size = os.path.getsize(file_path)
                            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            html += f"""
                            <tr>
                                <td><a href="{directory}/{file}">{file}</a></td>
                                <td>{size:,} bytes</td>
                                <td>{mtime.strftime('%Y-%m-%d %H:%M')}</td>
                            </tr>
                            """
                        except:
                            html += f"""
                            <tr>
                                <td>{file}</td>
                                <td>N/A</td>
                                <td>N/A</td>
                            </tr>
                            """
                    
                    html += "</table>"
        
        html += """
            <hr>
            <p>Seismo Framework v1.0.0</p>
        </body>
        </html>
        """
        
        # Save index
        index_path = os.path.join(self.base_dir, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return index_path

# Utility function
def organize_project_reports():
    """Organize reports in the current project."""
    manager = ReportManager(base_dir="seismo_framework/reports")
    
    print("🔧 Organizing Seismo Framework reports...")
    
    # Organize existing reports
    manager.organize_existing_reports(".")
    
    # Clean up old reports
    manager.cleanup_old_reports(days_to_keep=90)
    
    # Generate index
    index_path = manager.generate_index()
    
    # Print summary
    summary = manager.get_report_summary()
    
    print("\n📊 Report Organization Complete:")
    print(f"   Total reports: {summary['total_reports']}")
    print(f"   Total size: {summary['total_size_mb']} MB")
    print(f"   Index file: {index_path}")
    
    print("\n📁 Directory Structure:")
    for dir_name, count in summary['by_directory'].items():
        print(f"   {dir_name}: {count} reports")
    
    return manager
