import pandas as pd
import pm4py

def DomesticDeclarations_cleaning(url):
    df = pm4py.read_xes(url)
    df = pm4py.format_dataframe(
        df,
        case_id='case:concept:name',
        activity_key='concept:name',
        timestamp_key='time:timestamp'
    )

    def transform_unknowns():
        s1 = df['case:DeclarationNumber']
        s2 = pd.Series(index=s1.index, dtype=s1.dtype)

        instance = 0
        for i in range(len(s1)):
            if s1[i] == 'UNKNOWN':
                s2[i] = str(f'UNKNOWN {instance}')
                if i + 1 < len(s1) and s1[i + 1] != 'UNKNOWN':
                    instance += 1
            else:
                s2.loc[i] = s1.loc[i]
        return s2

    df['case:DeclarationNumber'] = transform_unknowns()

    df = df[['case:Amount', 'org:role', 'time:timestamp', 'org:resource', 'concept:name', 'case:DeclarationNumber', 'case:concept:name']]

    trace_start = df.groupby('case:DeclarationNumber')['time:timestamp'].transform('min')

    # Time intervals in seconds
    df['time_prev'] = (df.groupby('case:DeclarationNumber')['time:timestamp']
                       .diff()
                       .dt.total_seconds()
                       .fillna(0))
    
    df['time_start'] = (df['time:timestamp'] - trace_start).dt.total_seconds()
    
    # Target converted directly to days
    trace_end = df.groupby('case:DeclarationNumber')['time:timestamp'].transform('max')
    df['remaining_time (y)'] = (trace_end - df['time:timestamp']).dt.total_seconds() / 86400.0

    df['step_counter'] = df.groupby('case:DeclarationNumber').cumcount()
    df['day of week'] = df['time:timestamp'].apply(lambda s_: s_.weekday())

    df = df[['case:Amount', 'org:role', 'day of week',
             'org:resource', 'concept:name', 'case:DeclarationNumber',
             'time_prev', 'time_start', 'step_counter', 'case:concept:name',
             'remaining_time (y)']]

    return df.reset_index(drop=True)


def main():
    url = '/Users/laimis/Documents/Dissertation/BPI_Challenge_2020/Training for Dissertation/Data/DomesticDeclarations.xes.gz'
    cleaned_df = DomesticDeclarations_cleaning(url)

    cleaned_df.to_csv('data/cleaned_data.csv', index=False)
    print("Data successfully cleaned and saved!")

if __name__ == '__main__':
    main()