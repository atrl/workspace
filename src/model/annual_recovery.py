"""Date-only selection of explicitly qualified annual recovery classifiers."""
import pandas as pd


class AnnualRecoveryControllers:
    def __init__(self, models, training_audit):
        self.models=dict(models)
        if training_audit.year.duplicated().any():raise ValueError('duplicate recovery audit year')
        self.audit=training_audit.set_index('year').to_dict('index')
        expected={int(y) for y,a in self.audit.items() if a['status']=='fitted'}
        if set(self.models)!=expected:raise ValueError('recovery models and qualified audit years differ')
        for year,a in self.audit.items():
            if a['status'] not in ['fitted','needs_data']:raise ValueError('unknown recovery training status')
            if a['status']=='fitted':
                if int(a['rows'])<int(a['minimum_rows']) or not 0<int(a['positive_rows'])<int(a['rows']):raise ValueError('unqualified recovery training sample')
                if not pd.Timestamp(a['last_label_available_at'])<pd.Timestamp(a['cutoff']):raise ValueError('recovery training label crosses cutoff')

    def __call__(self, trade_date):
        day=pd.Timestamp(trade_date)
        if pd.isna(day) or day.tzinfo is not None or day!=day.normalize():raise ValueError('local trading date required')
        if day.year not in self.audit:raise ValueError('needs_data: no declared recovery policy for year')
        row=self.audit[day.year]
        decision=day.tz_localize('America/New_York')+pd.Timedelta(hours=10,minutes=31)
        if pd.Timestamp(row['cutoff'])>decision:raise ValueError('annual recovery model not yet available')
        return {'frozen_open_action_value_model':None,'frozen_recovery_model':self.models.get(day.year)}
