"""
实验报告生成器

生成Markdown/HTML格式的实验报告。
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class ReportGenerator:
    """实验报告生成器"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path('./output/reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            'title': '水下图像增强+全景分割系统实验报告',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': []
        }

    def add_section(self, title: str, content: str, level: int = 2):
        self.data['sections'].append({'title': title, 'content': content, 'level': level})

    def add_metrics(self, title: str, metrics: Dict[str, Any]):
        self.data['sections'].append({'type': 'metrics', 'title': title, 'metrics': metrics})

    def generate_markdown(self) -> str:
        lines = [f"# {self.data['title']}\n", f"**时间**: {self.data['date']}\n", "---\n"]
        for section in self.data['sections']:
            if section.get('type') == 'metrics':
                lines.append(f"## {section['title']}\n")
                lines.append("| 指标 | 值 |")
                lines.append("|------|-----|")
                for k, v in section['metrics'].items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")
            else:
                prefix = "#" * section['level']
                lines.append(f"{prefix} {section['title']}\n")
                lines.append(f"{section['content']}\n")
        return "\n".join(lines)

    def save(self, filename: str) -> Path:
        content = self.generate_markdown()
        save_path = self.output_dir / f"{filename}.md"
        save_path.write_text(content, encoding='utf-8')
        return save_path


if __name__ == '__main__':
    print("ReportGenerator module loaded")
