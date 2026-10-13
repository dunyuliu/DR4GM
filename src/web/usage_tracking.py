"""Usage analytics for the DR4GM Interactive Explorer.

Split out of dr4gm_interactive_explorer.py (PATHWAY_FORWARD.md row 17) with
no behavior change -- UsageEvent/UsageTracker moved verbatim.

PATHWAY_FORWARD.md rows 27/31: these best-effort analytics sinks (local log
file, Supabase backup, email webhook) must not block the user's session on
failure -- but swallowing with a bare except Exception: pass hid every
failure with no trace at all. They still swallow and continue (same
fire-and-forget design), just not silently.
"""

import streamlit as st
import json
import datetime
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

_analytics_logger = logging.getLogger(__name__ + ".analytics")

# Usage tracking data structure
@dataclass
class UsageEvent:
    session_id: str
    timestamp: str
    event_type: str
    details: dict


class UsageTracker:
    """Simple usage tracking for analytics with GitHub integration"""
    
    def __init__(self):
        self.session_id = self._get_session_id()
        self.events = []
        self.log_file = Path("usage_analytics.log")
        self.batch_events = []  # Buffer for batch commits
        self.batch_size = 10  # Commit every 10 events
        
    def _get_session_id(self):
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def track_event(self, event_type: str, details: dict = None):
        """Track a usage event with comprehensive analytics data"""
        if details is None:
            details = {}
            
        # Add comprehensive user context
        enhanced_details = details.copy()
        enhanced_details.update(self._get_user_context())
        
        # Add performance metrics
        enhanced_details.update(self._get_performance_metrics())
        
        event = UsageEvent(
            session_id=self.session_id,
            timestamp=datetime.datetime.now().isoformat(),
            event_type=event_type,
            details=enhanced_details
        )
        
        self.events.append(event)
        self.batch_events.append(event)
        self._save_event(event)
        
        # Update behavioral tracking
        self._update_behavioral_metrics(event_type)
        
        # Flush the in-memory batch buffer once full (no git side effects —
        # see _clear_batch / PROJECT_RULES.md rule 5)
        if len(self.batch_events) >= self.batch_size:
            self._clear_batch()
    
    def _get_performance_metrics(self):
        """Get performance and technical metrics"""
        metrics = {}
        
        try:
            # Memory usage (if available)
            import psutil
            process = psutil.Process()
            metrics['memory_usage_mb'] = round(process.memory_info().rss / 1024 / 1024, 2)
            metrics['cpu_percent'] = round(process.cpu_percent(), 2)
        except ImportError:
            pass
        
        # Streamlit performance metrics
        if hasattr(st, 'session_state'):
            metrics['cache_hits'] = len(getattr(st.session_state, 'cache_hits', []))
            metrics['rerun_count'] = getattr(st.session_state, 'rerun_count', 0)
            
        # Loading time estimation
        if 'page_load_start' not in st.session_state:
            st.session_state.page_load_start = datetime.datetime.now()
            metrics['page_load_time'] = 0
        else:
            load_time = (datetime.datetime.now() - st.session_state.page_load_start).total_seconds()
            metrics['page_load_time'] = round(load_time, 2)
        
        return metrics
    
    def _update_behavioral_metrics(self, event_type):
        """Update behavioral tracking metrics"""
        if 'behavioral_metrics' not in st.session_state:
            st.session_state.behavioral_metrics = {
                'page_views': 0,
                'clicks': 0,
                'downloads': 0,
                'interactions': 0,
                'time_on_page': 0,
                'unique_features_used': set(),
                'last_activity': datetime.datetime.now(),
                'bounce_detected': False
            }
        
        metrics = st.session_state.behavioral_metrics
        now = datetime.datetime.now()
        
        # Update metrics based on event type (skip page_view)
        if 'click' in event_type.lower():
            metrics['clicks'] += 1
            metrics['interactions'] += 1
        elif 'download' in event_type.lower():
            metrics['downloads'] += 1
            metrics['interactions'] += 1
        else:
            metrics['interactions'] += 1
            
        # Track unique features
        metrics['unique_features_used'].add(event_type)
        
        # Calculate time between interactions
        if 'last_activity' in metrics:
            time_diff = (now - metrics['last_activity']).total_seconds()
            metrics['time_on_page'] += time_diff
            
            # Detect potential bounce (single page view, no interactions for 30+ seconds)
            if metrics['page_views'] == 1 and metrics['interactions'] == 1 and time_diff > 30:
                metrics['bounce_detected'] = True
        
        metrics['last_activity'] = now
        st.session_state.behavioral_metrics = metrics
    
    def track_page_view(self, page_name: str = 'main'):
        """Track page view with standard analytics data"""
        # Increment page view counter
        if 'page_views' not in st.session_state:
            st.session_state.page_views = 0
        st.session_state.page_views += 1
        
        self.track_event('page_view', {
            'page_name': page_name,
            'page_view_number': st.session_state.page_views,
            'is_returning_visitor': st.session_state.page_views > 1
        })
    
    def track_interaction(self, element_type: str, element_id: str, action: str = 'click'):
        """Track user interactions with UI elements"""
        self.track_event('interaction', {
            'element_type': element_type,
            'element_id': element_id,
            'action': action,
            'interaction_sequence': getattr(st.session_state, 'events_count', 0)
        })
    
    def track_performance_issue(self, issue_type: str, details: dict = None):
        """Track performance issues and errors"""
        if details is None:
            details = {}
            
        self.track_event('performance_issue', {
            'issue_type': issue_type,
            'error_details': details,
            'user_impact': 'high' if 'error' in issue_type.lower() else 'medium'
        })
    
    def _get_user_context(self):
        """Get comprehensive user context following standard analytics practices"""
        import hashlib
        import platform
        context = {}
        
        try:
            # === STANDARD ANALYTICS DATA ===
            
            # 1. IP & Network Information
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                headers = st.context.headers
                
                # Raw IP addresses (standard practice)
                context['ip_address'] = headers.get('X-Forwarded-For', headers.get('X-Real-IP', 'unknown'))
                context['user_agent'] = headers.get('User-Agent', 'unknown')
                context['host'] = headers.get('Host', 'unknown')
                context['referer'] = headers.get('Referer', 'direct')
                
                # Create anonymous user hash (persistent across sessions)
                ip_raw = context['ip_address']
                if ip_raw != 'unknown':
                    context['user_hash'] = hashlib.sha256(ip_raw.encode()).hexdigest()[:16]
                else:
                    context['user_hash'] = 'unknown'
                
                # Geographic inference (standard)
                context['accept_language'] = headers.get('Accept-Language', 'unknown')
                context['accept_encoding'] = headers.get('Accept-Encoding', 'unknown')
                
            # 2. Device & Browser Information (parsed from User-Agent)
            user_agent = context.get('user_agent', 'unknown')
            if user_agent != 'unknown':
                # Parse browser info (standard analytics)
                if 'Chrome' in user_agent:
                    context['browser'] = 'Chrome'
                elif 'Firefox' in user_agent:
                    context['browser'] = 'Firefox'
                elif 'Safari' in user_agent:
                    context['browser'] = 'Safari'
                elif 'Edge' in user_agent:
                    context['browser'] = 'Edge'
                else:
                    context['browser'] = 'Other'
                
                # Operating System detection
                if 'Windows' in user_agent:
                    context['os'] = 'Windows'
                elif 'Macintosh' in user_agent or 'Mac OS' in user_agent:
                    context['os'] = 'macOS'
                elif 'Linux' in user_agent:
                    context['os'] = 'Linux'
                elif 'iPhone' in user_agent:
                    context['os'] = 'iOS'
                elif 'Android' in user_agent:
                    context['os'] = 'Android'
                else:
                    context['os'] = 'Other'
                
                # Device type
                if any(mobile in user_agent for mobile in ['Mobile', 'Android', 'iPhone']):
                    context['device_type'] = 'Mobile'
                elif 'Tablet' in user_agent or 'iPad' in user_agent:
                    context['device_type'] = 'Tablet'
                else:
                    context['device_type'] = 'Desktop'
            
            # 3. Session Information (standard)
            now = datetime.datetime.now()
            context['timestamp'] = now.isoformat()
            context['utc_timestamp'] = now.utctimetuple()
            
            # Session tracking
            if 'session_start_time' not in st.session_state:
                st.session_state.session_start_time = now.isoformat()
                st.session_state.page_views = 0
                st.session_state.events_count = 0
                
            # Calculate session duration
            session_start = datetime.datetime.fromisoformat(st.session_state.session_start_time)
            context['session_duration'] = (now - session_start).total_seconds()
            context['session_start'] = st.session_state.session_start_time
            
            # Update counters
            st.session_state.events_count += 1
            context['session_events_count'] = st.session_state.events_count
            
            # 4. Platform Information
            context['deployment'] = 'streamlit_cloud' if 'streamlit.app' in context.get('host', '') else 'local'
            context['user_type'] = 'developer' if 'localhost' in context.get('host', '') else 'production_user'
            
            # 5. Screen/Viewport Information (if available via JavaScript)
            # Note: Limited in Streamlit, but we can infer from user agent
            if 'Mobile' in user_agent:
                context['estimated_screen_width'] = '390'  # Common mobile width
            else:
                context['estimated_screen_width'] = '1920'  # Common desktop width
                
        except Exception as e:
            # Fallback context with minimal data
            context = {
                'user_agent': 'unknown',
                'user_type': 'unknown',
                'deployment': 'unknown',
                'browser': 'unknown',
                'os': 'unknown',
                'device_type': 'unknown',
                'error': str(e)
            }
        
        return context
    
    def _save_event(self, event: UsageEvent):
        """Save event to multiple destinations for zero data loss"""
        event_data = asdict(event)
        
        # 1. Save to local file (works locally)
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            _analytics_logger.warning(f"Failed to write usage event to {self.log_file}: {e}")
        
        # 2. Send to Supabase (immediate, reliable)
        self._save_to_supabase(event_data)
        
        # 3. Backup to Google Sheets (redundant backup)
        self._save_to_google_sheets(event_data)
        
        # 4. Send email notifications for important events
        self._send_email_notification(event_data)
    
    def _save_to_supabase(self, event_data):
        """Save to Supabase database - optional backup storage"""
        # Supabase is optional - only try if configured
        try:
            import requests
            
            # Get Supabase credentials from Streamlit secrets
            supabase_url = st.secrets.get("supabase_url", "")
            supabase_key = st.secrets.get("supabase_key", "")
            
            if not supabase_url or not supabase_key:
                return  # No credentials configured - this is fine
            
            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            }
            
            # Insert into usage_events table
            response = requests.post(
                f"{supabase_url}/rest/v1/usage_events",
                headers=headers,
                json=event_data,
                timeout=5
            )

        except Exception as e:
            # Don't disrupt user experience, but don't hide it either.
            _analytics_logger.warning(f"Supabase usage-event backup failed: {e}")
    
    def _save_to_google_sheets(self, event_data):
        """Send to Google Sheets via Apps Script webhook"""
        try:
            # Get webhook URL from Streamlit secrets (handle both local and cloud)
            webhook_url = ""
            
            # Try different ways to get the webhook URL
            try:
                webhook_url = st.secrets["google_sheets_webhook"]
            except (KeyError, AttributeError):
                try:
                    webhook_url = st.secrets.get("google_sheets_webhook", "")
                except:
                    # Fallback - check if secrets exist at all
                    try:
                        # In cloud environment, secrets might be structured differently
                        if hasattr(st.secrets, '_file_paths') or hasattr(st.secrets, '_secrets'):
                            webhook_url = getattr(st.secrets, 'google_sheets_webhook', "")
                    except:
                        pass
            
            if not webhook_url:
                # Only show error in debug mode - don't spam users
                if st.session_state.get('debug_analytics', False):
                    st.error("🚨 DEBUG: No Google Sheets webhook configured in Streamlit secrets!")
                return
            
            import requests
            
            # Flatten the data for Google Sheets
            flat_data = {
                'timestamp': event_data['timestamp'],
                'session_id': event_data['session_id'],
                'event_type': event_data['event_type'],
                'details': json.dumps(event_data['details'])  # Convert dict to JSON string
            }
            
            # Send to Google Apps Script webhook
            response = requests.post(
                webhook_url,
                json=flat_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            # Only show debug info if debug mode is enabled
            if st.session_state.get('debug_analytics', False):
                if response.status_code == 200:
                    st.success(f"✅ DEBUG: Google Sheets responded: {response.text[:100]}")
                else:
                    st.error(f"❌ DEBUG: Google Sheets error: {response.status_code}")
            
        except Exception as e:
            # Only show error in debug mode
            if st.session_state.get('debug_analytics', False):
                st.error(f"🚨 DEBUG: Google Sheets error: {str(e)}")
            pass
    
    def _send_email_notification(self, event_data):
        """Send email notification on first user action"""
        try:
            # Initialize action counter if not exists
            if 'total_user_actions' not in st.session_state:
                st.session_state.total_user_actions = 0
            
            # Increment total actions counter
            st.session_state.total_user_actions += 1
            
            # Only send email on the very first action
            if st.session_state.total_user_actions != 1:
                return
            
            # Get email settings from secrets
            email_webhook = st.secrets.get("email_webhook", "")
            email_to = st.secrets.get("notification_email", "")
            
            if not email_webhook or not email_to:
                return
            
            import requests
            
            # Create email content for new session
            details = event_data.get('details', {})
            user_type = details.get('user_type', 'unknown')
            deployment = details.get('deployment', 'unknown')
            
            # Determine if this is you (developer) or a real user
            is_developer = (user_type == 'developer' or 'localhost' in details.get('host', ''))
            user_emoji = '👨‍💻 [YOU]' if is_developer else '👤 [NEW USER]'
            
            subject = f"🌋 DR4GM First Action: {user_emoji}"
            
            body = f"""
🌋 DR4GM Interactive Explorer - First User Action

{user_emoji} **{'Developer Action' if is_developer else 'REAL USER ACTION!'}**

📅 Time: {event_data['timestamp']}
👤 Session: {event_data['session_id'][:8]}...
🎯 First Action: {event_data['event_type']}
🌐 Source: {deployment} ({details.get('host', 'unknown')})
🔍 Browser: {details.get('browser', 'unknown')} on {details.get('os', 'unknown')}
📱 Device: {details.get('device_type', 'unknown')}
📊 Total Actions: {st.session_state.total_user_actions}

{'' if is_developer else '🎉 **REAL USER IS INTERACTING!** Someone is actively using your DR4GM tool!'}

---
This is an automated notification from your DR4GM usage tracking system.
View complete analytics: [Google Sheets]
            """
            
            # Send via webhook (works with Zapier, IFTTT, etc.)
            email_data = {
                'to': email_to,
                'subject': subject,
                'body': body,
                'event_type': event_data['event_type'],
                'timestamp': event_data['timestamp']
            }
            
            response = requests.post(
                email_webhook,
                json=email_data,
                timeout=5
            )

        except Exception as e:
            # Don't disrupt user experience, but don't hide it either.
            _analytics_logger.warning(f"Email-notification webhook failed: {e}")
    
    def _clear_batch(self):
        """Clear the batch buffer once events have been locally persisted.

        PROJECT_RULES.md rule 5: production code under src/ must never
        `git add`/`commit`/`push` against the repo it is running in. This
        method previously did exactly that, autonomously, on a timer,
        wrapped in a bare `except: pass`. Removed, not relocated — events
        are already durably written per-event by `_save_event` (see
        `track_event`); batching here only bounds in-memory buffer size.
        """
        self.batch_events = []

    def force_commit(self):
        """Flush any pending batched events (no git side effects)."""
        if self.batch_events:
            self._clear_batch()

    def get_session_stats(self):
        """Get current session statistics"""
        return {
            'session_id': self.session_id,
            'events_count': len(self.events),
            'event_types': list(set(event.event_type for event in self.events)),
            'pending_commits': len(self.batch_events)
        }
    
    def get_all_usage_data(self):
        """Retrieve all usage data from Supabase"""
        try:
            import requests
            
            supabase_url = st.secrets.get("supabase_url", "")
            supabase_key = st.secrets.get("supabase_key", "")
            
            if not supabase_url or not supabase_key:
                return []
            
            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}"
            }
            
            response = requests.get(
                f"{supabase_url}/rest/v1/usage_events?order=timestamp.desc&limit=1000",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception:
            return []
