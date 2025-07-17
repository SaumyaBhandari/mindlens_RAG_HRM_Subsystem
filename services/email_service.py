import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.email_from = os.getenv("EMAIL_FROM")
        self.team_emails = os.getenv("EMAIL_TO")

    async def _send_email(self, recipient_email: str, subject: str, body: str):
        try:
            message = MIMEMultipart()
            message["From"] = self.email_from
            message["To"] = recipient_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            await aiosmtplib.send(
                message,
                hostname=self.smtp_server,
                port=self.smtp_port,
                start_tls=True,
                username=self.smtp_username,
                password=self.smtp_password
            )
            print(f"Email successfully sent to {recipient_email}")
        except Exception as e:
            import traceback
            print(f"Error sending email to {recipient_email}: {e}")
            traceback.print_exc()

    async def send_candidate_confirmation_email(self, full_name: str, email: str, interview_date: str, interview_time: str):
        subject = f"Interview Confirmation - {full_name}"
        body = f"""
                    Dear {full_name},

                    This email confirms your interview has been scheduled with the following details:

                    Date: {interview_date}
                    Time: {interview_time}

                    We look forward to speaking with you.

                    Best regards,
                    The Interview Team
            """
        await self._send_email(email, subject, body)

    async def send_team_notification_email(self, full_name: str, email: str, interview_date: str, interview_time: str):
        if not self.team_emails:
            print("No team email recipients configured. Skipping team notification.")
            return

        subject = f"New Interview Scheduled, and Task Completion Email - {full_name}"
        body = f"""
                Dear Interviewer,

                The candidate {full_name} has completed the task assigned to him.
                Here is the link of github where the task has been completed:
                https://github.com/SaumyaBhandari/mindlens_RAG_HRM_Subsystem

                A new interview has been booked with the following details:
                
                Candidate Name: {full_name}
                Candidate Email: {email}
                Interview Date: {interview_date}
                Interview Time: {interview_time}

                Please make some adjustments according to your needs, and you can send him an email
                invitation for his interview as a Senior ML Engineer Role.

                Best regards,
                Saumya Bhandary's Palmmind's RAG System
                """

        await self._send_email(email, subject, body)

    async def send_interview_notifications(self, full_name: str, email: str, interview_date: str, interview_time: str):
        print(f"Attempting to send interview notifications for {full_name}...")
        await self.send_candidate_confirmation_email(full_name, email, interview_date, interview_time)
        await self.send_team_notification_email(full_name, self.team_emails, interview_date, interview_time)
        print(f"Finished attempting to send interview notifications for {full_name}.")