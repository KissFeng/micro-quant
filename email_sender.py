#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email 推送模块 - 发送分析报告到邮箱
"""

import os
import smtplib
import base64
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

from config import EMAIL_CONFIG


class EmailSender:
    """Email 发送器"""

    def __init__(self, smtp_server=None, smtp_port=None, sender=None, password=None, recipient=None):
        """
        初始化 Email 发送器

        参数:
            smtp_server: SMTP 服务器地址
            smtp_port: SMTP 端口
            sender: 发件人邮箱
            password: 邮箱授权码
            recipient: 收件人邮箱
        """
        config = EMAIL_CONFIG
        self.smtp_server = smtp_server or config['smtp_server']
        self.smtp_port = smtp_port or config['smtp_port']
        self.sender = sender or config['sender']
        self.password = password or config['password']
        self.recipient = recipient or config['recipient']

    def send_email(self, subject, html_content, attachments=None):
        """
        发送邮件

        参数:
            subject: 邮件主题
            html_content: HTML 格式的邮件内容
            attachments: 附件路径列表

        返回:
            bool: 发送是否成功
        """
        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = self.recipient
        msg['Subject'] = subject
        msg['Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 添加 HTML 内容
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        # 添加附件
        if attachments:
            for filepath in attachments:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    filename = os.path.basename(filepath)
                    part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                    msg.attach(part)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 使用 SSL 连接
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, self.recipient, msg.as_string())
                print(f"✓ 邮件已发送: {subject}")
                # 发送成功后等待，避免连续发送被限流
                time.sleep(5)
                return True
            except smtplib.SMTPServerDisconnected as e:
                print(f"✗ 邮件发送失败(第{attempt+1}次): {e}")
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  等待{wait}秒后重试...")
                    time.sleep(wait)
            except Exception as e:
                print(f"✗ 邮件发送失败: {e}")
                return False
        return False

    def send_prediction_report(self, predictions_df, model_name='Model', date=None):
        """
        发送预测报告邮件

        参数:
            predictions_df: 预测结果 DataFrame
            model_name: 模型名称
            date: 预测日期
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        subject = f"[量化系统] 预测报告 - {model_name} - {date}"

        # 构建 HTML 内容
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                .buy {{ color: green; font-weight: bold; }}
                .sell {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h2>📊 量化系统 - 预测报告</h2>
            <p><strong>模型:</strong> {model_name}</p>
            <p><strong>日期:</strong> {date}</p>
            <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="summary">
                <h3>📈 预测统计</h3>
                <p>股票数量: {len(predictions_df)}</p>
                <p>预测上涨: {(predictions_df['probability'] > 0.5).sum()} 只</p>
                <p>预测下跌: {(predictions_df['probability'] <= 0.5).sum()} 只</p>
                <p>平均上涨概率: {predictions_df['probability'].mean():.2%}</p>
            </div>

            <h3>📋 详细预测</h3>
            <table>
                <tr>
                    <th>股票代码</th>
                    <th>股票名称</th>
                    <th>预测概率</th>
                    <th>预测方向</th>
                    <th>置信度</th>
                </tr>
        """

        for _, row in predictions_df.iterrows():
            prob = row['probability']
            direction = "📈 上涨" if prob > 0.5 else "📉 下跌"
            confidence = "高" if abs(prob - 0.5) > 0.15 else ("中" if abs(prob - 0.5) > 0.05 else "低")
            css_class = "buy" if prob > 0.5 else "sell"

            html += f"""
                <tr>
                    <td>{row.get('stock', 'N/A')}</td>
                    <td>{row.get('name', row.get('stock', 'N/A'))}</td>
                    <td>{prob:.2%}</td>
                    <td class="{css_class}">{direction}</td>
                    <td>{confidence}</td>
                </tr>
            """

        html += """
            </table>
            <p style="margin-top: 30px; color: #7f8c8d; font-size: 12px;">
                此邮件由量化系统自动发送，请勿回复。
            </p>
        </body>
        </html>
        """

        # 发送邮件（不带附件）
        self.send_email(subject, html)

    def send_backtest_report(self, results_dict, chart_path=None):
        """
        发送回测报告邮件

        参数:
            results_dict: 回测结果字典
            chart_path: 图表路径
        """
        date = datetime.now().strftime('%Y-%m-%d')
        subject = f"[量化系统] 回测报告 - {date}"

        # 构建 HTML 内容
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2 {{ color: #2c3e50; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
                .metric-box {{ background-color: #3498db; color: white; padding: 15px; border-radius: 5px; min-width: 150px; text-align: center; }}
                .metric-box.green {{ background-color: #27ae60; }}
                .metric-box.red {{ background-color: #e74c3c; }}
                .metric-box.orange {{ background-color: #f39c12; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 12px; margin-top: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #2c3e50; color: white; }}
            </style>
        </head>
        <body>
            <h2>📊 量化系统 - 回测报告</h2>
            <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h3>📈 关键指标</h3>
            <div class="metrics">
                <div class="metric-box {'green' if results_dict.get('total_return', 0) > 0 else 'red'}">
                    <div class="metric-value">{results_dict.get('total_return', 0)*100:.2f}%</div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{results_dict.get('annual_return', 0)*100:.2f}%</div>
                    <div class="metric-label">年化收益率</div>
                </div>
                <div class="metric-box orange">
                    <div class="metric-value">{results_dict.get('sharpe_ratio', 0):.3f}</div>
                    <div class="metric-label">夏普比率</div>
                </div>
                <div class="metric-box red">
                    <div class="metric-value">{results_dict.get('max_drawdown', 0)*100:.2f}%</div>
                    <div class="metric-label">最大回撤</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{results_dict.get('win_rate', 0)*100:.1f}%</div>
                    <div class="metric-label">胜率</div>
                </div>
            </div>

            <h3>📋 交易统计</h3>
            <table>
                <tr><td>总交易次数</td><td>{results_dict.get('total_trades', 0)}</td></tr>
                <tr><td>盈利交易</td><td>{results_dict.get('winning_trades', 0)}</td></tr>
                <tr><td>最终资金</td><td>HK${results_dict.get('final_capital', 0):,.2f}</td></tr>
                <tr><td>基准收益</td><td>{results_dict.get('benchmark_return', 0)*100:.2f}%</td></tr>
            </table>

            <p style="margin-top: 30px; color: #7f8c8d; font-size: 12px;">
                此邮件由量化系统自动发送，请勿回复。
            </p>
        </body>
        </html>
        """

        # 发送邮件（带图表附件）
        attachments = [chart_path] if chart_path and os.path.exists(chart_path) else None
        self.send_email(subject, html, attachments)

    def send_model_comparison_report(self, results_df):
        """
        发送多模型对比报告邮件

        参数:
            results_df: 对比结果 DataFrame
        """
        date = datetime.now().strftime('%Y-%m-%d')
        subject = f"[量化系统] 模型对比报告 - {date}"

        # 找出最佳模型
        best_idx = results_df['auc'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .best {{ background-color: #27ae60 !important; color: white; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h2>📊 量化系统 - 多模型对比报告</h2>
            <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h3>📈 模型评估结果</h3>
            <table>
                <tr>
                    <th>模型</th>
                    <th>准确率</th>
                    <th>AUC</th>
                    <th>精确率</th>
                    <th>召回率</th>
                    <th>F1分数</th>
                </tr>
        """

        for _, row in results_df.iterrows():
            is_best = row['Model'] == best_model
            css_class = 'class="best"' if is_best else ''
            html += f"""
                <tr {css_class}>
                    <td>{row['Model']} {'⭐' if is_best else ''}</td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['auc']:.4f}</td>
                    <td>{row['precision']:.4f}</td>
                    <td>{row['recall']:.4f}</td>
                    <td>{row['f1']:.4f}</td>
                </tr>
            """

        html += f"""
            </table>

            <div style="background-color: #d5f4e6; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <h3>🏆 推荐模型</h3>
                <p><strong>{best_model}</strong> 在 AUC 指标上表现最优，推荐用于实际预测。</p>
            </div>

            <p style="margin-top: 30px; color: #7f8c8d; font-size: 12px;">
                此邮件由量化系统自动发送，请勿回复。
            </p>
        </body>
        </html>
        """

        self.send_email(subject, html)


if __name__ == '__main__':
    print("测试 Email 发送模块...")
    print("注意：请先配置 config.py 中的 EMAIL_CONFIG")

    # 简单测试（需要网络和正确配置）
    # sender = EmailSender()
    # sender.send_prediction_report(pd.DataFrame({
    #     'stock': ['000001.SZ'],
    #     'name': ['平安银行'],
    #     'probability': [0.62]
    # }), 'LightGBM')
