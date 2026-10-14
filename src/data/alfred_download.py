"""Anonymous official ALFRED initial-release downloads with bounded form batches."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import zipfile
import pandas as pd
import requests
from bs4 import BeautifulSoup


def download_initial_releases(series, directory, *, start, as_of):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    url='https://alfred.stlouisfed.org/series/downloaddata?seid='+series
    session=requests.Session();response=session.get(url,timeout=30);response.raise_for_status()
    soup=BeautifulSoup(response.content,'html.parser')
    form=next(f for f in soup.find_all('form') if f.find(attrs={'name':'form[file_type]'}) is not None)
    begin=max(pd.Timestamp(start),pd.Timestamp(form.find(attrs={'name':'form[obs_start_date]'}).get('value')))
    end=min(pd.Timestamp(as_of),pd.Timestamp(form.find(attrs={'name':'form[obs_end_date]'}).get('value')))
    vintage=[o.get('value') for o in form.find(attrs={'name':'form[selected_vintage_dates][]'}).find_all('option')]
    vintage=[d for d in vintage if pd.Timestamp(start)-pd.Timedelta(days=7)<=pd.Timestamp(d)<=pd.Timestamp(as_of)]
    if not vintage:raise ValueError('no official vintage dates in requested interval')
    entries=[];parts=[]
    for batch,offset in enumerate(range(0,len(vintage),400)):
        dates=vintage[offset:offset+400];path=directory/f'batch_{batch:02d}.zip'
        if not path.exists():
            data=[('form[units]','lin'),('form[obs_start_date]',str(begin.date())),('form[obs_end_date]',str(end.date())),('form[entered_vintage_dates]',''),('form[file_type]','4'),('form[file_format]','csv'),('form[download_data]','')]
            data += [('form[selected_vintage_dates][]',d) for d in dates]
            data += [(h.get('name'),h.get('value','')) for h in form.find_all('input',attrs={'type':'hidden'})]
            result=session.post(url,data=data,timeout=45);result.raise_for_status()
            if not zipfile.is_zipfile(io.BytesIO(result.content)):raise ValueError('official download did not return a ZIP')
            path.write_bytes(result.content)
        raw=path.read_bytes();archive=zipfile.ZipFile(io.BytesIO(raw));readme=archive.read('README.txt').decode()
        if 'Series ID: '+series not in readme or 'Output Format: Observations, Initial Release Only' not in readme:raise ValueError('archive identity/format differs')
        names=[n for n in archive.namelist() if n.endswith('.csv')]
        if len(names)!=1:raise ValueError('unexpected archive contents')
        frame=pd.read_csv(io.BytesIO(archive.read(names[0])))
        if list(frame)!=['period_start_date',series,'realtime_start_date']:raise ValueError('initial-release schema differs')
        frame=frame.rename(columns={'period_start_date':'observation_date',series:'value','realtime_start_date':'release_date'})
        for name in ['observation_date','release_date']:frame[name]=pd.to_datetime(frame[name])
        if not frame.release_date.dt.strftime('%Y-%m-%d').isin(dates).all():raise ValueError('archive releases outside requested vintage batch')
        if not frame.observation_date.between(begin,end).all():raise ValueError('archive observations outside requested interval')
        frame['value']=pd.to_numeric(frame.value,errors='coerce');frame['archive_sha256']=hashlib.sha256(raw).hexdigest();parts.append(frame)
        entries.append({'path':path.name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'csv_name':names[0],'vintage_dates':dates,'rows':len(frame)})
    combined=pd.concat(parts,ignore_index=True).sort_values('observation_date')
    if combined.observation_date.duplicated().any():raise ValueError('duplicate initial-release observations across batches')
    if combined[['observation_date','release_date']].isna().any().any() or combined.observation_date.gt(as_of).any() or combined.release_date.gt(as_of).any():raise ValueError('unknown or after-cutoff archive dates')
    manifest={'series_id':series,'source_url':url,'start':start,'as_of':as_of,'output_format':'Observations, Initial Release Only','retrieved_at':datetime.now(timezone.utc).isoformat(),'first_selected_vintage':vintage[0],'last_selected_vintage':vintage[-1],'entries':entries,'first_archive_is_not_original_source_publication':True}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return combined,manifest
