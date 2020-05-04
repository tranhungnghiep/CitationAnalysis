"""
Prepare data for high impact paper prediction.
Data source: Microsoft Academic Graph (MAG)

Outline:
- compute feature for x:
    - paper:
        - author count
        - ref count
        - title length
        - published month: for each paper, not venue; jour may publish many times a year.
        - domain diversity based on ref.
        - domain diversity based on keyword.
        - paper title/keyword/field (topic or embeddings...).
        - novelty: KL on word distribution of paper title vs ref title.
        - domain of paper: important.
    - author:
        - num of coauthors
        - career length
        - paper count
        - last year paper count
        - cit count
        - last year cit count
        - h index
        - h5 index
        - impact factor in last year y: (cit in y of paper pub in y-1 and y-2)/(num of paper pub in y-1 and y-2)
        - domain diversity: entropy = - Sum_over_all_domain( paper in each domain / paper in all domain * log(paper in each domain / paper in all domain) )
        - paper title/keyword/field.
    - venue:
        - venue run length
        - paper count
        - last year paper count
        - citcount
        - last year cit count
        - h index
        - h5 index
        - impact factor in last year y: (cit in y of paper pub in y-1 and y-2)/(num of paper pub in y-1 and y-2)
        - is journal or is conference.
        - domain diversity.
        - paper title/keyword/field.
        - domain of venue: important.
- compute y:
    - sum citcount
    - ranking in each venue
    - top k: k is param.

Note:
- In MAG, paper file: conf and jour have 2 separate columns, so their id could be duplicate. Have to make unify venue id.
"""


import os
import argparse
import time

import pandas as pd
import numpy as np
np.random.seed(7)
import random
random.seed(7)

from prepare_data_citcount import unify_venue_np, eval_cols


def main(args):

    print('START.')
    start_time_main = time.time()  # in second.


    compute_simplefeature(args)


    print('FINISH.')
    stop_time_main = time.time()
    print('Time (s): ' + str(stop_time_main-start_time_main))

    return 0


def compute_simplefeature(args):
    """Compute simple features from baseline.
    Algorithm:
        - read
        - compute
    """

    global debug

    # Working folder.
    root_path = args.root_path

    start_test_year = args.start_test_year
    end_test_year = args.end_test_year
    period = args.period  # used for venue paper rank list.
    min_year_step = 1
    max_year_step = period  # temporarily used for computing citcount sequence.
    min_year = args.min_year
    max_year = end_test_year + period  # 2005


    # 1. Read csv files.
    print('READ FILES.')
    start_time = time.time()  # in second.

    paper_filename = 'Papers.txt'
    names_paper = ['Paper ID', 'Orig title', 'Norm title', 'Year', 'Date', 'DOI', 'Venue name', 'Norm venue name', 'Journal ID', 'Conference ID', 'Rank']
    dtype_paper = {'Paper ID': str, 'Orig title': str, 'Norm title': str, 'Year': int, 'Date': str, 'DOI': str, 'Venue name': str, 'Norm venue name': str, 'Journal ID': str, 'Conference ID': str, 'Rank': int}
    usecols_paper = ['Paper ID', 'Orig title', 'Year', 'Date', 'Journal ID', 'Conference ID']
    parse_dates_paper = ['Date']
    paper_paper_filename = 'PaperReferences.txt'
    names_paper_paper = ['Paper ID', 'Paper ref ID']
    usecols_paper_paper = ['Paper ID', 'Paper ref ID']
    paper_author_filename = 'PaperAuthorAffiliations.txt'
    names_paper_author = ['Paper ID', 'Author ID', 'Affil ID', 'Orig affil name', 'Norm affil name', 'Author sequence number']
    usecols_paper_author = ['Paper ID', 'Author ID']

    paper_keyword_filename = 'PaperKeywords.txt'
    names_paper_keyword = ['Paper ID', 'Keyword name', 'Domain ID']
    usecols_paper_keyword = ['Paper ID', 'Domain ID']
    domain_filename = 'FieldsOfStudy.txt'
    names_domain = ['Domain ID', 'Domain name']
    usecols_domain = ['Domain ID', 'Domain name']
    domain_level_filename = 'FieldOfStudyHierarchy.txt'
    names_domain_level = ['Child ID', 'Child level', 'Parent ID', 'Parent level', 'Confidence']
    usecols_domain_level = ['Child ID', 'Child level', 'Parent ID', 'Parent level']

    paper_file = os.path.join(root_path, paper_filename)
    paper_paper_file = os.path.join(root_path, paper_paper_filename)
    paper_author_file = os.path.join(root_path, paper_author_filename)

    paper_keyword_file = os.path.join(root_path, paper_keyword_filename)
    domain_file = os.path.join(root_path, domain_filename)
    domain_level_file = os.path.join(root_path, domain_level_filename)

    nrows = None
    if args.debug_server:
        if 'rows' in args.config:
            nrows = int(args.config[0])
        if 'nodate' in args.config:
            parse_dates_paper = False

    # Sequential reading files:
    base_time = time.time()
    paper = pd.read_csv(paper_file, delimiter='\t', header=None, skiprows=0, names=names_paper, usecols=usecols_paper, dtype=dtype_paper, engine='c', nrows=nrows)
    print('\t' + 'Read paper: ' + str(time.time() - base_time)); base_time = time.time()
    if parse_dates_paper is not False:
        for date_col in parse_dates_paper:
            paper.loc[:, date_col] = pd.to_datetime(paper.loc[:, date_col], format='%Y/%m/%d', errors='coerce')
    print('\t' + 'Parse date in paper: ' + str(time.time() - base_time)); base_time = time.time()
    paper_paper = pd.read_csv(paper_paper_file, delimiter='\t', header=None, skiprows=0, names=names_paper_paper, usecols=usecols_paper_paper, dtype=str, engine='c', nrows=nrows)
    print('\t' + 'Read paper paper: ' + str(time.time() - base_time)); base_time = time.time()
    paper_author = pd.read_csv(paper_author_file, delimiter='\t', header=None, skiprows=0, names=names_paper_author, usecols=usecols_paper_author, dtype=str, engine='c', nrows=nrows)
    print('\t' + 'Read paper author: ' + str(time.time() - base_time)); base_time = time.time()
    paper_keyword = pd.read_csv(paper_keyword_file, delimiter='\t', header=None, skiprows=0, names=names_paper_keyword, usecols=usecols_paper_keyword, dtype=str, engine='c', nrows=nrows)
    print('\t' + 'Read paper keyword: ' + str(time.time() - base_time)); base_time = time.time()
    domain = pd.read_csv(domain_file, delimiter='\t', header=None, skiprows=0, names=names_domain, usecols=usecols_domain, dtype=str, engine='c')
    print('\t' + 'Read domain: ' + str(time.time() - base_time)); base_time = time.time()
    domain_level = pd.read_csv(domain_level_file, delimiter='\t', header=None, skiprows=0, names=names_domain_level, usecols=usecols_domain_level, dtype=str, engine='c')
    print('\t' + 'Read domain level: ' + str(time.time() - base_time)); base_time = time.time()

    print('READ FILES: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


    # 2. Filter by domain 'Computer Science'.
    print('FILTER BY DOMAIN.')
    start_time = time.time()  # in second.

    if debug:
        # First, for sanity check, save all domain level 0.
        domain_level0 = domain.loc[domain.loc[:, 'Domain ID'].isin(domain_level.loc[domain_level.loc[:, 'Parent level'].values == 'L0', 'Parent ID'].drop_duplicates().values), :] \
            .sort_values('Domain name').reset_index(drop=True)  # isin() is faster than merge: 17ms vs 44ms.
        domain_level0.to_csv(os.path.join(root_path, args.save_dir, 'domain_level0.csv'))

    # Confirmed, domain level 0 'Computer Science', Domain ID = 0271BC14.
    domain_cs = pd.DataFrame(['0271BC14'], columns=['Domain ID'])
    while True:
        domain_cs_temp = domain_level.loc[domain_level.loc[:, 'Parent ID'].isin(domain_cs.loc[:, 'Domain ID'].values), ['Parent ID', 'Child ID']]  # isin() is faster than merge: 14ms vs 25ms.
        domain_cs_temp = pd.DataFrame(np.concatenate((domain_cs_temp.loc[:, ['Parent ID']].values, domain_cs_temp.loc[:, ['Child ID']].values)), columns=['Domain ID']).drop_duplicates()
        print('\t' + 'Check domain level complete: ' + str(domain_cs.shape == domain_cs_temp.shape and (domain_cs.values == domain_cs_temp.values).all().all()))
        if domain_cs.shape == domain_cs_temp.shape and (domain_cs.values == domain_cs_temp.values).all().all():  # Note that comparison with None or NaN is always False.
            break
        else:
            domain_cs = domain_cs_temp

    # Then filter paper in CS.
    paper_id = paper_keyword.loc[paper_keyword.loc[:, 'Domain ID'].isin(domain_cs.loc[:, 'Domain ID'].values), ['Paper ID']].drop_duplicates()  # isin: 17ms, merge: 19 ms.
    paper = paper.loc[paper.loc[:, 'Paper ID'].isin(paper_id.loc[:, 'Paper ID'].values), :]  # isin: 3ms, merge: 18ms.

    print('FILTER BY DOMAIN: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


    # 3. Add UNIFY VENUE column.
    print('UNIFY VENUE.')
    start_time = time.time()  # in second.

    j = paper.columns.get_loc("Journal ID")
    c = paper.columns.get_loc("Conference ID")
    paper.loc[:, 'Unify venue'] = [unify_venue_np(row, j, c) for row in paper.values]
    paper = paper.loc[:, ['Paper ID', 'Orig title', 'Year', 'Date', 'Unify venue']]

    print('UNIFY VENUE: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


    # 4. Filter 'noisy' paper.
    # ONLY USE PAPERS FROM 1970 TO 2010 (TRIMMED TO 2005, based on actually used papers), WITH NOT NULL TITLE, VENUE, AUTHOR. (KEYWORD, DOMAIN filtered above).
    print('FILTER NOISY PAPER AND BUILD PAPER ALIGN.')
    start_time = time.time()  # in second.

    # Convert to numpy array before comparing to produce no-label masking arrays, use bitwise '&' to merge 2 logical array. Note that comparison to None or NaN will result in False.
    paper = paper.loc[min_year <= paper.loc[:, 'Year'].values, :]
    paper = paper.loc[paper.loc[:, 'Year'].values <= max_year, :]  # Separate loc is faster than bitwise &. Only use bitwise when |.
    paper = paper.loc[pd.notnull(paper.loc[:, 'Orig title'].values), :]
    paper = paper.loc[pd.notnull(paper.loc[:, 'Unify venue'].values), :]
    paper_align = paper.merge(paper_author, left_on='Paper ID', right_on='Paper ID')  # This filters both paper and author: paper must have author, author's paper must be clean. Must use merge, cannot use isin, because need data from both df.
    eval_cols(paper_align, ['Paper ID', 'Orig title', 'Year', 'Date', 'Unify venue', 'Author ID'])  # CHECKPOINT.
    paper_align = paper_align.drop_duplicates()  # Just to make sure.

    print('FILTER NOISY PAPER AND BUILD PAPER ALIGN: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    # Trim paper_paper by filtered papers (1970-2005, Computer Science, no noise).
    print('TRIM PAPER_PAPER BY PAPER_ALIGN.')
    start_time = time.time()  # in second.

    paper_paper = paper_paper.loc[paper_paper.loc[:, 'Paper ID'].isin(paper_align.loc[:, 'Paper ID'].values), :]  # Note: paper_paper is only in CS
    paper_paper = paper_paper.loc[paper_paper.loc[:, 'Paper ref ID'].isin(paper_align.loc[:, 'Paper ID'].values), :]  # isin: 33ms, merge: 55ms. Separate loc is faster than bitwise. Only need to use bitwise |, for bitwise & better use separate loc.
    paper_paper = paper_paper.drop_duplicates()  # Just to make sure.

    print('TRIM PAPER_PAPER BY PAPER_ALIGN: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    # Trim paper_align by paper_paper.
    print('TRIM PAPER_ALIGN BY PAPER_PAPER.')
    start_time = time.time()  # in second.

    paper_align = paper_align.loc[paper_align.loc[:, 'Paper ID'].isin(paper_paper.loc[:, ['Paper ID', 'Paper ref ID']].values.ravel()), :]  # isin(series.values) is bit faster than isin(series). large.isin(small) is a bit faster than small.isin(large), but the resulting boolean mask is very large (same size as the large series, use lots of ram). Order is important, it has the meaning, hereby trimming paper_align after trimming paper_paper, meaning prioritizing cutting all noise and scaling down.

    print('TRIM PAPER_ALIGN BY PAPER_PAPER: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time - start_time))

    print('After cleaning, get back to unique paper.')
    paper = paper_align.loc[:, ['Paper ID', 'Orig title', 'Year', 'Date', 'Unify venue']].drop_duplicates()  # This is the unique paper, do not need to drop duplicate from paper align all the time.

    print('Active data are only upto year ' + str(end_test_year))
    active_paper_align = paper_align.loc[paper_align.loc[:, 'Year'].values <= end_test_year, :]
    active_paper_paper = paper_paper.loc[paper_paper.loc[:, 'Paper ref ID'].isin(active_paper_align.loc[:, 'Paper ID'].values), :]
    active_paper = paper.loc[paper.loc[:, 'Year'].values <= end_test_year, :]

    print('\t' + '# Paper_align nodes: ' + str(len(paper_align.loc[:, 'Paper ID'].unique())))  # CHECKPOINT.
    print('\t' + '# Paper_paper nodes: ' + str(len(pd.unique(paper_paper.loc[:, ['Paper ID', 'Paper ref ID']].values.ravel()))))  # CHECKPOINT.
    print('\t' + '# Paper nodes: ' + str(paper.shape[0]))  # CHECKPOINT.
    print('\t' + '# Active Paper_align nodes: ' + str(len(active_paper_align.loc[:, 'Paper ID'].unique())))  # CHECKPOINT.
    print('\t' + '# Active Paper_paper nodes: ' + str(len(pd.unique(active_paper_paper.loc[:, ['Paper ID', 'Paper ref ID']].values.ravel()))))  # CHECKPOINT.
    print('\t' + '# Active Paper nodes: ' + str(active_paper.shape[0]))  # CHECKPOINT.


    # 5. Extract paper align. Just save. Notice the correct column order.
    if args.save_align:
        print('SAVE PAPER ALIGN.')
        start_time = time.time()  # in second.

        paper_align.loc[:, ['Year', 'Paper ID', 'Author ID', 'Unify venue']].to_csv(os.path.join(root_path, args.save_dir, 'PAPER_ALIGN_'+str(min_year)+'_'+str(max_year)+'.txt'), sep=' ', header=False, index=False)

        print('SAVE PAPER ALIGN: DONE.')
        stop_time = time.time()
        print('Time (s): ' + str(stop_time-start_time))


    # 6.1 Extract cit count sequence. And save. Rerun, last time wrong.
    if args.save_citcount:
        # Citation count.
        print('SAVE CITATION COUNT SEQUENCE.')
        start_time = time.time()  # in second.

        citation_count = active_paper_paper \
            .merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID', right_on='Paper ID') \
            .merge(active_paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ref ID', right_on='Paper ID')  # Must use merge, cannot use isin, because need data from both df.
        eval_cols(citation_count, ['Paper ID'+'_x', 'Paper ref ID', 'Year'+'_x', 'Paper ID'+'_y', 'Year'+'_y'])  # CHECKPOINT. Merge on different column names, but no new 'key_0'?
        citation_count.loc[:, 'Year step'] = citation_count.loc[:, 'Year'+'_x'].values - citation_count.loc[:, 'Year'+'_y'].values
        citation_count = citation_count.loc[min_year_step <= citation_count.loc[:, 'Year step'].values, :]
        citation_count = citation_count.loc[citation_count.loc[:, 'Year step'].values <= max_year_step, :]  # loc is faster than bitwise &.
        citation_count = citation_count.groupby(['Paper ID'+'_y', 'Year step'], sort=False).size().reset_index(name='Size')  # How many citation to paper y == the number of rows in the group of paper y.
        eval_cols(citation_count, ['Paper ID'+'_y', 'Year step', 'Size'])  # CHECKPOINT.
        citation_count = citation_count.rename(columns={'Paper ID'+'_y': 'Paper ID'})
        citation_count = citation_count.drop_duplicates()  # Just to make sure.
        citation_count.to_csv(os.path.join(root_path, args.save_dir, 'CITATION_COUNT_'+str(min_year)+'_'+str(end_test_year)+'.txt'), sep=' ', header=False, index=False)
        citation_count = None

        print('SAVE CITATION COUNT SEQUENCE: DONE.')
        stop_time = time.time()
        print('Time (s): ' + str(stop_time-start_time))


    # 6.2 Extract venue paper rank. And save.
    if args.save_venue_paper_rank:
        print('SAVE VENUE PAPER RANK IN THE PERIOD.')
        start_time = time.time()  # in second.

        citation_count = active_paper_paper \
            .merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID', right_on='Paper ID') \
            .merge(active_paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ref ID', right_on='Paper ID')  # Must use merge, cannot use isin, because need data from both df.
        eval_cols(citation_count, ['Paper ID'+'_x', 'Paper ref ID', 'Year'+'_x', 'Paper ID'+'_y', 'Year'+'_y'])  # CHECKPOINT. Merge on different column names, but no new 'key_0'?
        citation_count = citation_count.loc[citation_count.loc[:, 'Year'+'_x'].values - citation_count.loc[:, 'Year'+'_y'].values <= period, :]  # Filter.
        citation_count = citation_count.groupby(['Paper ID'+'_y', 'Year'+'_y'], sort=False).size().reset_index(name='Size')  # How many citation to paper y == the number of rows in the group of paper y.
        eval_cols(citation_count, ['Paper ID'+'_y', 'Year'+'_y', 'Size'])  # CHECKPOINT. Paper y publish in year y has size citations after period 5 years.
        citation_count = citation_count.rename(columns={'Paper ID'+'_y': 'Paper ID', 'Year'+'_y': 'Year'})
        # Add venue info.
        citation_count = citation_count \
            .merge(active_paper.loc[:, ['Paper ID', 'Unify venue']], on='Paper ID')
        eval_cols(citation_count, ['Paper ID', 'Year', 'Size', 'Unify venue'])  # CHECKPOINT.
        citation_count.loc[:, 'Rank'] = citation_count.sort_values(['Unify venue', 'Year', 'Size', 'Paper ID'], ascending=[True, True, False, True]) \
            .groupby(['Unify venue', 'Year'], sort=False).cumcount() + 1  # groupby preserves order no need to sort Unify venue year but sort for clarity, cumcount return order as series, matched with citation_count by index.
        citation_count = citation_count.sort_values(['Unify venue', 'Year', 'Rank'])
        eval_cols(citation_count, ['Paper ID', 'Year', 'Size', 'Unify venue', 'Rank'])  # CHECKPOINT. Just for info.
        citation_count = citation_count.drop_duplicates()  # Just to make sure.
        citation_count.loc[:, ['Unify venue', 'Year', 'Paper ID', 'Size', 'Rank']].to_csv(os.path.join(root_path, args.save_dir, 'VENUE_PAPER_RANK_PERIOD'+str(period)+'.txt'), sep=' ', header=True, index=False)
        citation_count = None

        print('SAVE VENUE PAPER RANK IN THE PERIOD: DONE.')
        stop_time = time.time()
        print('Time (s): ' + str(stop_time-start_time))


    # 7. No network, compute simple features.
    print('COMPUTE SIMPLE FEATURES.')
    start_time = time.time()  # in second.

    # Note: just compute each simple feature, then combine by merging based on paper_align.

    # Paper.
    base_time = time.time()
    paper_simple = active_paper.loc[:, ['Paper ID']]
    print('\t' + 'Paper simple time: ' + str(time.time() - base_time)); base_time = time.time()

    # author count
    author_count = active_paper_align.groupby(['Paper ID'], sort=False).size().reset_index(name='Author count')
    print('\t' + '\t' + 'Author count time: ' + str(time.time() - base_time)); base_time = time.time()

    # ref count
    ref_count = active_paper_paper.groupby(['Paper ID'], sort=False).size().reset_index(name='Ref count')  # some papers do not have ref info
    print('\t' + '\t' + 'Ref count time: ' + str(time.time() - base_time)); base_time = time.time()

    # title length
    title_length = active_paper.loc[:, ['Paper ID', 'Orig title']]
    title_length.loc[:, 'Title len'] = [len(title.split()) if pd.notnull(title) else 0 for title in title_length.loc[:, 'Orig title'].values]
    print('\t' + '\t' + 'Title len time: ' + str(time.time() - base_time)); base_time = time.time()

    # pub month: to use in sklearn: OneHotEncoder transform.
    pub_month = active_paper.loc[:, ['Paper ID', 'Date']]
    pub_month.loc[:, 'Month'] = [date.month if pd.notnull(date) and type(date) == pd.tslib.Timestamp else 0 for date in pub_month.loc[:, 'Date']]  # if there's no pub date: use month 0.
    print('\t' + '\t' + 'Pub month time: ' + str(time.time() - base_time)); base_time = time.time()

    if args.debug_server and 'date' in args.config:
        print(pub_month.loc[:, 'Date'].head(20))

    # Merge:
    paper_simple = paper_simple \
        .merge(author_count, on=['Paper ID'], how='left') \
        .merge(ref_count, on=['Paper ID'], how='left') \
        .merge(title_length.loc[:, ['Paper ID', 'Title len']], on=['Paper ID'], how='left') \
        .merge(pub_month.loc[:, ['Paper ID', 'Month']], on=['Paper ID'], how='left')
    paper_simple = paper_simple.fillna(0)
    paper_simple.to_csv(os.path.join(root_path, args.save_dir, 'PAPER_SIMPLE' + '.txt'), sep=' ', header=True, index=False)

    if args.debug_server and 'unittest' in args.config:
        utest = active_paper.iloc[np.random.randint(0, active_paper.shape[0], 10), :].drop_duplicates()
        utest.merge(paper_simple, on=['Paper ID'], how='left') \
            .to_csv(os.path.join(root_path, args.temp_dir, 'Unittest_' + 'PAPER_SIMPLE' + '.txt'), sep=' ', header=True, index=False)
        utest.merge(paper_paper, on=['Paper ID'], how='left') \
            .to_csv(os.path.join(root_path, args.temp_dir, 'Unittest_' + 'PAPER_SIMPLE' + '_ref' + '.txt'), sep=' ', header=True, index=False)

        utest = active_paper_align.iloc[np.random.randint(0, active_paper_align.shape[0], 10), :].loc[:, ['Author ID']].drop_duplicates()
        utest.merge(paper_align, on=['Author ID'], how='left') \
            .merge(paper_align, on=['Paper ID'], how='left').sort_values(['Author ID'+'_x', 'Year'+'_x', 'Author ID'+'_y']) \
            .to_csv(os.path.join(root_path, args.temp_dir, 'Unittest_' + 'AUTHOR_SIMPLE' + '_coauthor' + '.txt'), sep=' ', header=True, index=False)
        utest.merge(paper_align, on=['Author ID'], how='left').sort_values(['Author ID', 'Year']) \
            .to_csv(os.path.join(root_path, args.temp_dir, 'Unittest_' + 'AUTHOR_SIMPLE' + '_careerlen_pcount' + '.txt'), sep=' ', header=True, index=False)
        utest.merge(paper_align, on=['Author ID'], how='left') \
            .merge(paper_paper, left_on=['Paper ID'], right_on=['Paper ref ID'], how='left') \
            .merge(paper, left_on=['Paper ID'+'_y'], right_on=['Paper ID'], how='left').sort_values(['Author ID', 'Year'+'_y', 'Year'+'_x']) \
            .to_csv(os.path.join(root_path, args.temp_dir, 'Unittest_' + 'AUTHOR_SIMPLE' + '_ccount' + '.txt'), sep=' ', header=True, index=False)

        utest = active_paper.iloc[np.random.randint(0, active_paper.shape[0], 10), :].loc[:, ['Unify venue']].drop_duplicates()
        utest = utest.append({'Unify venue': 'C.4390334E'}, ignore_index=True)
        utest.merge(paper, on=['Unify venue'], how='left').sort_values(['Unify venue', 'Year']) \
            .to_csv(os.path.join(root_path, args.temp_dir, 'Unittest_' + 'VENUE_SIMPLE' + '_runlen_pcount' + '.txt'), sep=' ', header=True, index=False)
        utest.merge(paper, on=['Unify venue'], how='left') \
            .merge(paper_paper, left_on=['Paper ID'], right_on=['Paper ref ID'], how='left') \
            .merge(paper, left_on=['Paper ID'+'_y'], right_on=['Paper ID'], how='left').sort_values(['Unify venue'+'_x', 'Year'+'_y', 'Year'+'_x']) \
            .to_csv(os.path.join(root_path, args.temp_dir, 'Unittest_' + 'VENUE_SIMPLE' + '_ccount' + '.txt'), sep=' ', header=True, index=False)

        # RESULT:
        # Paper: ok
        # Author: ok
        # Venue: ok
        return 0

    paper_simple = None
    author_count = None
    ref_count = None
    title_length = None
    pub_month = None
    print('\t' + 'Paper simple: Save time: ' + str(time.time() - base_time)); base_time = time.time()


    # Author.
    author_simple = active_paper_align.loc[:, ['Author ID', 'Year']].drop_duplicates()
    print('\t' + 'Author simple time: ' + str(time.time() - base_time)); base_time = time.time()

    # num coauthor (distinct) upto last year
    # alg: merge on paper to have author-paper-year-coauthor, then sort by year to drop duplicate coauthor keep first, then count coauthor each year, then cumsum upto each year. When read feature, need to post process to have the correct count for each year.
    num_coauthor = active_paper_align.loc[:, ['Author ID', 'Paper ID', 'Year']].merge(active_paper_align.loc[:, ['Author ID', 'Paper ID']], on=['Paper ID'])  # same paper, same year
    eval_cols(num_coauthor, ['Author ID'+'_x', 'Paper ID', 'Year', 'Author ID'+'_y'])
    num_coauthor = num_coauthor.loc[num_coauthor.loc[:, 'Author ID'+'_x'].values != num_coauthor.loc[:, 'Author ID'+'_y'].values, :]  # coauthor have to be another author
    num_coauthor = num_coauthor.sort_values(['Author ID'+'_x', 'Year']).drop_duplicates(['Author ID'+'_x', 'Author ID'+'_y'], keep='first')  # each coauthor counted only once, in temporal order. May lose some later years here. Do not care about paper id
    # Count coauthor.
    num_coauthor = num_coauthor.groupby(['Author ID'+'_x', 'Year'], sort=False).size().reset_index(name='Coauthor count')
    num_coauthor = num_coauthor.rename(columns={'Author ID'+'_x': 'Author ID'})
    # Get back all lost years (possibly authors) because of drop duplicate.
    num_coauthor = num_coauthor.merge(author_simple, on=['Author ID', 'Year'], how='outer')  # need to outer merge to get back all lost years before cumsum
    eval_cols(num_coauthor, ['Author ID', 'Year', 'Coauthor count'])
    num_coauthor.loc[:, 'Coauthor count'] = num_coauthor.loc[:, 'Coauthor count'].fillna(0)
    # Sum upto each year.
    num_coauthor.loc[:, 'Cum coauthor count'] = num_coauthor.sort_values(['Author ID', 'Year']).groupby(['Author ID'], sort=False)['Coauthor count'].cumsum()
    num_coauthor.loc[:, 'Cum coauthor count upto last year'] = num_coauthor.loc[:, 'Cum coauthor count'].values - num_coauthor.loc[:, 'Coauthor count'].values
    eval_cols(num_coauthor, ['Author ID', 'Year', 'Coauthor count', 'Cum coauthor count', 'Cum coauthor count upto last year'])
    print('\t' + '\t' + 'Coauthor time: ' + str(time.time() - base_time)); base_time = time.time()

    # career length
    # alg: for each author find first year, then merge on author id to have first year for each author year, then subtract to have career length.
    first_year = author_simple.groupby(['Author ID'], sort=False)['Year'].min().reset_index(name='First year')
    career_length = author_simple.merge(first_year, on='Author ID')
    eval_cols(career_length, ['Author ID', 'Year', 'First year'])
    career_length.loc[:, 'Career len'] = career_length.loc[:, 'Year'].values - career_length.loc[:, 'First year'].values
    print('\t' + '\t' + 'Career len time: ' + str(time.time() - base_time)); base_time = time.time()

    # paper count upto last year
    # alg: count paper of each author in each year, then sort by year and cum sum paper count upto each year, then substract paper count of current year to get cumsum upto last year.
    paper_count = active_paper_align.groupby(['Author ID', 'Year'], sort=False).size().reset_index(name='Paper count')
    paper_count.loc[:, 'Cum paper count'] = paper_count.sort_values(['Author ID', 'Year']).groupby(['Author ID'], sort=False)['Paper count'].cumsum()
    paper_count.loc[:, 'Cum paper count upto last year'] = paper_count.loc[:, 'Cum paper count'].values - paper_count.loc[:, 'Paper count'].values
    print('\t' + '\t' + 'Paper count time: ' + str(time.time() - base_time)); base_time = time.time()

    # last year paper count
    # alg: how to get value in previous row? Option 1: just for-loop over dataframe: O(n), reuse paper_count
    paper_count = paper_count.sort_values(['Author ID', 'Year'])
    paper_count.loc[:, 'Paper count last year'] = 0  # add column, default 0
    eval_cols(paper_count, ['Author ID', 'Year', 'Paper count', 'Cum paper count', 'Cum paper count upto last year', 'Paper count last year'])
    # Option 2: pandas trick: use df.loc[].shift(k) to shift k rows, or .diff(k) to get the difference between rows; can also do this in group, with df.grroupby()[col].shift/diff
    paper_count.loc[paper_count.groupby(['Author ID'], sort=False)['Year'].diff(1) == 1, 'Paper count last year'] = \
        paper_count.groupby(['Author ID'], sort=False)['Paper count'].shift(1)  # must use series: left and right is different size, use series to match index
    print('\t' + '\t' + 'Last year paper count time: ' + str(time.time() - base_time)); base_time = time.time()

    # cit count upto last year
    # alg: merge author-paper with paper-paper on paper ref id to have author-paper-citingpaper-citingyear, then count cite in each citing year, then outer merge with author simple to have all pub year, then cumsum upto each year both cit year and pub year, then subtract current cit count to get cumsum to last year.
    cit_count = active_paper_align.loc[:, ['Author ID', 'Paper ID']].merge(active_paper_paper, left_on='Paper ID', right_on='Paper ref ID')  # may smaller than author simple if some papers have no citation
    eval_cols(cit_count, ['Author ID', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID'])  # paper id x == paper ref id, paper id y is citing paper
    cit_count = cit_count.merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID'+'_y', right_on='Paper ID')  # get year of citing paper
    eval_cols(cit_count, ['Author ID', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID', 'Paper ID', 'Year'])  # paper id y == paper id: citing paper
    # Count citation of each author received in each year.
    cit_count = cit_count.groupby(['Author ID', 'Year'], sort=False).size().reset_index(name='Cit count')
    # Get back all lost author-year because of merging with paper_paper before cumsum.
    cit_count = cit_count.merge(author_simple, on=['Author ID', 'Year'], how='outer')  # outer join: get new a column Year with all pub years and all cit years, get a new column Author ID with all author id of author simple (larger).
    eval_cols(cit_count, ['Author ID', 'Year', 'Cit count'])
    cit_count.loc[:, 'Cit count'] = cit_count.loc[:, 'Cit count'].fillna(0)
    # Sum up citation count to each year.
    cit_count.loc[:, 'Cum cit count'] = cit_count.sort_values(['Author ID', 'Year']).groupby(['Author ID'], sort=False)['Cit count'].cumsum()
    cit_count.loc[:, 'Cum cit count upto last year'] = cit_count.loc[:, 'Cum cit count'].values - cit_count.loc[:, 'Cit count'].values
    print('\t' + '\t' + 'Citcount time: ' + str(time.time() - base_time)); base_time = time.time()

    # last year cit count
    # alg: similar to last year paper count
    cit_count = cit_count.sort_values(['Author ID', 'Year'])
    cit_count.loc[:, 'Cit count last year'] = 0  # add column, default 0
    eval_cols(cit_count, ['Author ID', 'Year', 'Cit count', 'Cum cit count', 'Cum cit count upto last year', 'Cit count last year'])
    # Option 2: pandas trick: use df.loc[].shift(k) to shift k rows, or .diff(k) to get the difference between rows; can also do this in group, with df.grroupby()[col].shift/diff
    cit_count.loc[cit_count.groupby(['Author ID'], sort=False)['Year'].diff(1) == 1, 'Cit count last year'] = \
        cit_count.groupby(['Author ID'], sort=False)['Cit count'].shift(1)
    print('\t' + '\t' + 'Last year citcount time: ' + str(time.time() - base_time)); base_time = time.time()

    # IMPACT FACTOR in last year y: (cit in y of paper pub in y-1 and y-2)/(num of paper pub in y-1 and y-2)
    # alg: merge to have author-paper-year-citingpaper-citingyear, then filter to have only citingyear - year = {1, 2}, then count citing paper of each author each year, for counting paper need to count paper beforehand and outer join, then for-loop and compute IF.
    if_window = 2
    # First, for paper count each year, we already have paper_count above.
    # Second, for citation count, cannot use citation count each year, because that includes citation to any past papers, for IF we only count citation each year to papers before them 1 or 2 year.
    impact_factor = active_paper_align.loc[:, ['Author ID', 'Paper ID', 'Year']].merge(active_paper_paper, left_on='Paper ID', right_on='Paper ref ID')  # may smaller than author simple if some papers have no citation
    eval_cols(impact_factor, ['Author ID', 'Paper ID'+'_x', 'Year', 'Paper ID'+'_y', 'Paper ref ID'])  # paper id x == paper ref id, paper id y is citing paper
    impact_factor = impact_factor.merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID'+'_y', right_on='Paper ID')  # get year of citing paper
    eval_cols(impact_factor, ['Author ID', 'Paper ID'+'_x', 'Year'+'_x', 'Paper ID'+'_y', 'Paper ref ID', 'Paper ID', 'Year'+'_y'])  # paper id y == paper id, year x is of paper x and paper ref, year y is of citing paper
    # Filter before count. Note that we have pairs of cited paper - citing paper, we also only count these pairs, so filtering these pair is enough.
    impact_factor = impact_factor.loc[(impact_factor.loc[:, 'Year'+'_y'] - impact_factor.loc[:, 'Year'+'_x']) <= if_window, :]
    impact_factor = impact_factor.loc[1 <= (impact_factor.loc[:, 'Year'+'_y'] - impact_factor.loc[:, 'Year'+'_x']), :]
    # Count citation received each year.
    impact_factor = impact_factor.groupby(['Author ID', 'Year'+'_y'], sort=False).size().reset_index(name='Cit count back 2 years')
    impact_factor = impact_factor.rename(columns={'Year'+'_y': 'Year'})
    # Merge to have both paper count and cit count back 2 years. And retain only author-year in author_simple.
    impact_factor = impact_factor.merge(paper_count.loc[:, ['Author ID', 'Year', 'Paper count']], on=['Author ID', 'Year'], how='outer')  # keep all year paper count and year cit count, compute impact factor for all these years, although only care about impact factor in those years when there are published papers
    eval_cols(impact_factor, ['Author ID', 'Year', 'Cit count back 2 years', 'Paper count'])  # Note: year is only from paper_count, this also filters cit count, only care about cit count when there a paper count, that is only retain author-year in paper_count
    impact_factor.loc[:, ['Cit count back 2 years', 'Paper count']] = impact_factor.loc[:, ['Cit count back 2 years', 'Paper count']].fillna(0)  # be careful: assign series or df will be matched by column name and index
    # Compute IF. If do not have enough information (2 years) or 0 citation, compute partial IF with the available information (1 year).
    impact_factor = impact_factor.sort_values(['Author ID', 'Year'])
    # Option 2: pandas trick: use df.loc[].shift(k) to shift k rows, or .diff(k) to get the difference between rows; can also do this in group, with df.grroupby()[col].shift/diff
    # alg: add 1 column for sum pub in 2 last consecutive years of each author, if = cit/pub when pub != 0.
    impact_factor.loc[:, 'Pub 2 years back'] = 0
    impact_factor.loc[:, 'IF'] = 0  # add column, default 0
    for j in range(1, 1+if_window):
        impact_factor \
            .loc[impact_factor.groupby(['Author ID'], sort=False)['Year'].diff(j) <= if_window, 'Pub 2 years back'] = \
            impact_factor.loc[:, 'Pub 2 years back'] + impact_factor.groupby(['Author ID'], sort=False)['Paper count'].shift(j).fillna(0)
    impact_factor.loc[:, 'IF'] = np.true_divide(impact_factor.loc[:, 'Cit count back 2 years'].values, impact_factor.loc[:, 'Pub 2 years back'].values,
                                                out=np.zeros(impact_factor.shape[0]), where=impact_factor.loc[:, 'Pub 2 years back'].values!=0)
    impact_factor.loc[:, 'IF last year'] = 0  # add column, default 0
    impact_factor.loc[impact_factor.groupby(['Author ID'], sort=False)['Year'].diff(1) == 1, 'IF last year'] = \
        impact_factor.groupby(['Author ID'], sort=False)['IF'].shift(1)
    eval_cols(impact_factor, ['Author ID', 'Year', 'Cit count back 2 years', 'Paper count', 'Pub 2 years back', 'IF', 'IF last year'])
    print('\t' + '\t' + 'Impact factor time: ' + str(time.time() - base_time)); base_time = time.time()

    # H INDEX upto last year: redo.
    # For each year, h index is different. For each year, citation count of each paper need to be recomputed, rank need to be recomputed.
    # ONLY 2 WAYS:
    # 1: replicate paper to all coauthors and to all years of all each author's papers. Then group by each author, each year and compute h index.
    #   alg: need to replicate paper to all years before using this method to compute h index.
    #       => first, 1 route: merge to have author-paper-citingpaper-citingyear. We could count citation each year here.
    #       => then, other route: merge author-paper and author-year to have author-paper-allpubyear.
    #       ==> then collapse 2 route by outer merge, to have author-paper-bothciting&allpubyear-citcounteachyear.
    # 2: just count citation of each paper in some years, but need to loop over the list and compute h index manually.
    #   alg: merge to have author-paper-citing paper-citing paper year, then count citation of each paper in each year receiving citations, then cumsum to have citation count of each paper upto each citing year, then loop over author-year and get the data to determine h index.
    # 1st route:
    detail_cit_count = active_paper_align.loc[:, ['Author ID', 'Paper ID']].merge(active_paper_paper, left_on='Paper ID', right_on='Paper ref ID')  # may smaller than author simple if some papers have no citation
    eval_cols(detail_cit_count, ['Author ID', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID'])  # paper id x == paper ref id, paper y cites paper x
    detail_cit_count = detail_cit_count.merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID'+'_y', right_on='Paper ID')  # get year of citing paper
    eval_cols(detail_cit_count, ['Author ID', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID', 'Paper ID', 'Year'])  # paper id y == paper id: citing paper
    # Count citation of each paper received in each year (note that papers are repeated for all coauthors).
    detail_cit_count = detail_cit_count.groupby(['Author ID', 'Paper ID'+'_x', 'Year'], sort=False).size().reset_index(name='Cit count')
    detail_cit_count = detail_cit_count.rename(columns={'Paper ID'+'_x': 'Paper ID'})
    eval_cols(detail_cit_count, ['Author ID', 'Paper ID', 'Year', 'Cit count'])
    # 2nd route:
    detail_year = active_paper_align.loc[:, ['Author ID', 'Paper ID']] \
        .merge(active_paper_align.loc[:, ['Author ID', 'Year']].drop_duplicates(), on=['Author ID'])
    eval_cols(detail_year, ['Author ID', 'Paper ID', 'Year'])
    # collapse 2 routes:
    detail_cit_count = detail_cit_count.merge(detail_year, on=['Author ID', 'Paper ID', 'Year'], how='outer')
    eval_cols(detail_cit_count, ['Author ID', 'Paper ID', 'Year', 'Cit count'])
    detail_cit_count = detail_cit_count.fillna(0)
    # Sum citation count of EACH PAPER upto each CITING AND PUB year.
    detail_cit_count.loc[:, 'Cum cit count'] = detail_cit_count.sort_values(['Author ID', 'Paper ID', 'Year']).groupby(['Author ID', 'Paper ID'], sort=False)['Cit count'].cumsum()
    detail_cit_count.loc[:, 'Cum cit count upto last year'] = detail_cit_count.loc[:, 'Cum cit count'].values - detail_cit_count.loc[:, 'Cit count'].values
    eval_cols(detail_cit_count, ['Author ID', 'Paper ID', 'Year', 'Cit count', 'Cum cit count', 'Cum cit count upto last year'])
    # Filter h5:
    detail_cit_count = detail_cit_count.merge(active_paper.loc[:, ['Paper ID', 'Year']], on='Paper ID')  # get year of cited paper, no need to outer, only care about papers that have citation
    eval_cols(detail_cit_count, ['Author ID', 'Paper ID', 'Year'+'_x', 'Cit count', 'Cum cit count', 'Cum cit count upto last year', 'Year'+'_y'])
    detail_cit_count = detail_cit_count.rename(columns={'Year'+'_x': 'Year', 'Year'+'_y': 'Pub year'})
    h_window = 5
    detail_cit_count5 = detail_cit_count.loc[detail_cit_count.loc[:, 'Year'].values - detail_cit_count.loc[:, 'Pub year'].values <= h_window, :]  # publish year must be not too old with regards to the year to compute h index. No need to check lower bound, because already compute cum citcount upto last year.
    # Compute h index: alg: groupby author-year, sort by cum citcount desc (paper asc for deterministic), cumcount rank, select top h paper, group by author-year count paper as h index.
    detail_cit_count.loc[:, 'Rank'] = detail_cit_count.sort_values(['Author ID', 'Year', 'Cum cit count upto last year', 'Paper ID'], ascending=[True, True, False, True]) \
        .groupby(['Author ID', 'Year'], sort=False).cumcount() + 1
    detail_cit_count = detail_cit_count.loc[detail_cit_count.loc[:, 'Rank'].values <= detail_cit_count.loc[:, 'Cum cit count upto last year'].values, :]
    h_index = detail_cit_count.groupby(['Author ID', 'Year'], sort=False).size().reset_index(name='H index')
    # Compute h5 index:
    detail_cit_count5.loc[:, 'Rank'] = detail_cit_count5.sort_values(['Author ID', 'Year', 'Cum cit count upto last year', 'Paper ID'], ascending=[True, True, False, True]) \
        .groupby(['Author ID', 'Year'], sort=False).cumcount() + 1
    detail_cit_count5 = detail_cit_count5.loc[detail_cit_count5.loc[:, 'Rank'].values <= detail_cit_count5.loc[:, 'Cum cit count upto last year'].values, :]
    h5_index = detail_cit_count5.groupby(['Author ID', 'Year'], sort=False).size().reset_index(name='H5 index')
    print('\t' + '\t' + 'H index time: ' + str(time.time() - base_time)); base_time = time.time()

    # Merge:
    author_simple = author_simple \
        .merge(num_coauthor.loc[:, ['Author ID', 'Year', 'Cum coauthor count upto last year']], on=['Author ID', 'Year'], how='left') \
        .merge(career_length.loc[:, ['Author ID', 'Year', 'Career len']], on=['Author ID', 'Year'], how='left') \
        .merge(paper_count.loc[:, ['Author ID', 'Year', 'Cum paper count upto last year', 'Paper count last year']], on=['Author ID', 'Year'], how='left') \
        .merge(cit_count.loc[:, ['Author ID', 'Year', 'Cum cit count upto last year', 'Cit count last year']], on=['Author ID', 'Year'], how='left')\
        .merge(h_index.loc[:, ['Author ID', 'Year', 'H index']], on=['Author ID', 'Year'], how='left') \
        .merge(h5_index.loc[:, ['Author ID', 'Year', 'H5 index']], on=['Author ID', 'Year'], how='left') \
        .merge(impact_factor.loc[:, ['Author ID', 'Year', 'IF last year']], on=['Author ID', 'Year'], how='left')
    author_simple = author_simple.fillna(0)
    author_simple.to_csv(os.path.join(root_path, args.save_dir, 'AUTHOR_SIMPLE' + '.txt'), sep=' ', header=True, index=False)

    author_simple = None
    num_coauthor = None
    career_length = None
    paper_count = None
    cit_count = None
    impact_factor = None
    h_index = None
    h5_index = None
    detail_cit_count = None
    detail_cit_count5 = None
    detail_year = None
    first_year = None
    print('\t' + 'Author simple: Save time: ' + str(time.time() - base_time)); base_time = time.time()


    # Venue. Use similar alg as Author.
    venue_simple = active_paper.loc[:, ['Unify venue', 'Year']].drop_duplicates()
    print('\t' + 'Venue simple time: ' + str(time.time() - base_time)); base_time = time.time()

    # venue run length
    first_year = venue_simple.groupby(['Unify venue'], sort=False)['Year'].min().reset_index(name='First year')
    venue_runlength = venue_simple.merge(first_year, on='Unify venue')
    eval_cols(venue_runlength, ['Unify venue', 'Year', 'First year'])
    venue_runlength.loc[:, 'Run len'] = venue_runlength.loc[:, 'Year'].values - venue_runlength.loc[:, 'First year'].values
    print('\t' + '\t' + 'Venue run len time: ' + str(time.time() - base_time)); base_time = time.time()

    # paper count upto last year
    paper_count = active_paper.groupby(['Unify venue', 'Year'], sort=False).size().reset_index(name='Paper count')
    paper_count.loc[:, 'Cum paper count'] = paper_count.sort_values(['Unify venue', 'Year']).groupby(['Unify venue'], sort=False)['Paper count'].cumsum()
    paper_count.loc[:, 'Cum paper count upto last year'] = paper_count.loc[:, 'Cum paper count'].values - paper_count.loc[:, 'Paper count'].values
    print('\t' + '\t' + 'Paper count time: ' + str(time.time() - base_time)); base_time = time.time()

    # last year paper count
    paper_count = paper_count.sort_values(['Unify venue', 'Year'])
    paper_count.loc[:, 'Paper count last year'] = 0  # add column, default 0
    eval_cols(paper_count, ['Unify venue', 'Year', 'Paper count', 'Cum paper count', 'Cum paper count upto last year', 'Paper count last year'])
    # Option 2: pandas trick: use df.loc[].shift(k) to shift k rows, or .diff(k) to get the difference between rows; can also do this in group, with df.grroupby()[col].shift/diff
    paper_count.loc[paper_count.groupby(['Unify venue'], sort=False)['Year'].diff(1) == 1, 'Paper count last year'] = \
        paper_count.groupby(['Unify venue'], sort=False)['Paper count'].shift(1)
    print('\t' + '\t' + 'Last year paper count time: ' + str(time.time() - base_time)); base_time = time.time()

    # cit count upto last year
    cit_count = active_paper.loc[:, ['Unify venue', 'Paper ID']].merge(active_paper_paper, left_on='Paper ID', right_on='Paper ref ID')  # Different from author count: drop duplicates
    eval_cols(cit_count, ['Unify venue', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID'])  # paper id x == paper ref id, paper id y is citing paper
    cit_count = cit_count.merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID'+'_y', right_on='Paper ID')  # get year of citing paper
    eval_cols(cit_count, ['Unify venue', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID', 'Paper ID', 'Year'])  # paper id y == paper id: citing paper
    # Count citation of each Unify venue received in each year.
    cit_count = cit_count.groupby(['Unify venue', 'Year'], sort=False).size().reset_index(name='Cit count')
    # Get back all lost Unify venue-year because of merging with paper_paper before cumsum.
    cit_count = cit_count.merge(venue_simple, on=['Unify venue', 'Year'], how='outer')
    eval_cols(cit_count, ['Unify venue', 'Year', 'Cit count'])
    cit_count.loc[:, 'Cit count'] = cit_count.loc[:, 'Cit count'].fillna(0)
    # Sum up citation count to each year.
    cit_count.loc[:, 'Cum cit count'] = cit_count.sort_values(['Unify venue', 'Year']).groupby(['Unify venue'], sort=False)['Cit count'].cumsum()
    cit_count.loc[:, 'Cum cit count upto last year'] = cit_count.loc[:, 'Cum cit count'].values - cit_count.loc[:, 'Cit count'].values
    print('\t' + '\t' + 'Citcount time: ' + str(time.time() - base_time)); base_time = time.time()

    # last year cit count
    cit_count = cit_count.sort_values(['Unify venue', 'Year'])
    cit_count.loc[:, 'Cit count last year'] = 0  # add column, default 0
    eval_cols(cit_count, ['Unify venue', 'Year', 'Cit count', 'Cum cit count', 'Cum cit count upto last year', 'Cit count last year'])
    # Option 2: pandas trick: use df.loc[].shift(k) to shift k rows, or .diff(k) to get the difference between rows; can also do this in group, with df.grroupby()[col].shift/diff
    cit_count.loc[cit_count.groupby(['Unify venue'], sort=False)['Year'].diff(1) == 1, 'Cit count last year'] = \
        cit_count.groupby(['Unify venue'], sort=False)['Cit count'].shift(1)
    print('\t' + '\t' + 'Last year citcount time: ' + str(time.time() - base_time)); base_time = time.time()

    # IMPACT FACTOR in last year y: (cit in y of paper pub in y-1 and y-2)/(num of paper pub in y-1 and y-2)
    if_window = 2
    impact_factor = active_paper.loc[:, ['Unify venue', 'Paper ID', 'Year']].merge(active_paper_paper, left_on='Paper ID', right_on='Paper ref ID')  # Different from author count: drop duplicates
    eval_cols(impact_factor, ['Unify venue', 'Paper ID'+'_x', 'Year', 'Paper ID'+'_y', 'Paper ref ID'])  # paper id x == paper ref id, paper id y is citing paper
    impact_factor = impact_factor.merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID'+'_y', right_on='Paper ID')  # get year of citing paper
    eval_cols(impact_factor, ['Unify venue', 'Paper ID'+'_x', 'Year'+'_x', 'Paper ID'+'_y', 'Paper ref ID', 'Paper ID', 'Year'+'_y'])  # paper id y == paper id, year x is of paper x and paper ref, year y is of citing paper
    # Filter before count. Note that we have pairs of cited paper - citing paper, we also only count these pairs, so filtering these pair is enough. Note we have to filter before counting, unlike H index, because we count for venue not for each paper, so pub year is lost and we cannot filter later.
    impact_factor = impact_factor.loc[(impact_factor.loc[:, 'Year'+'_y'] - impact_factor.loc[:, 'Year'+'_x']) <= if_window, :]
    impact_factor = impact_factor.loc[1 <= (impact_factor.loc[:, 'Year'+'_y'] - impact_factor.loc[:, 'Year'+'_x']), :]
    # Count citation received each year.
    impact_factor = impact_factor.groupby(['Unify venue', 'Year'+'_y'], sort=False).size().reset_index(name='Cit count back 2 years')
    impact_factor = impact_factor.rename(columns={'Year'+'_y': 'Year'})
    # Merge to have both paper count and cit count back 2 years. And retain only Unify venue-year in venue_simple.
    impact_factor = impact_factor.merge(paper_count.loc[:, ['Unify venue', 'Year', 'Paper count']], on=['Unify venue', 'Year'], how='outer')  # keep all year paper count and year cit count, compute impact factor for all these years, although only care about impact factor in those years when there are published papers
    eval_cols(impact_factor, ['Unify venue', 'Year', 'Cit count back 2 years', 'Paper count'])  # Note: year is only from paper_count, this also filters cit count, only care about cit count when there a paper count, that is only retain author-year in paper_count
    impact_factor.loc[:, ['Cit count back 2 years', 'Paper count']] = impact_factor.loc[:, ['Cit count back 2 years', 'Paper count']].fillna(0)  # be careful: assign series or df will be matched by column name and index
    # Compute IF. If do not have enough information (2 years) or 0 citation, compute partial IF with the available information (1 year).
    impact_factor = impact_factor.sort_values(['Unify venue', 'Year'])
    # Option 2: pandas trick: use df.loc[].shift(k) to shift k rows, or .diff(k) to get the difference between rows; can also do this in group, with df.grroupby()[col].shift/diff
    # alg: add 1 column for sum pub in 2 last consecutive years of each Unify venue, if = cit/pub when pub != 0.
    impact_factor.loc[:, 'Pub 2 years back'] = 0
    impact_factor.loc[:, 'IF'] = 0  # add column, default 0
    for j in range(1, 1+if_window):
        impact_factor \
            .loc[impact_factor.groupby(['Unify venue'], sort=False)['Year'].diff(j) <= if_window, 'Pub 2 years back'] = \
            impact_factor.loc[:, 'Pub 2 years back'] + impact_factor.groupby(['Unify venue'], sort=False)['Paper count'].shift(j).fillna(0)
    impact_factor.loc[:, 'IF'] = np.true_divide(impact_factor.loc[:, 'Cit count back 2 years'].values, impact_factor.loc[:, 'Pub 2 years back'].values,
                                                out=np.zeros(impact_factor.shape[0]), where=impact_factor.loc[:, 'Pub 2 years back'].values!=0)
    impact_factor.loc[:, 'IF last year'] = 0  # add column, default 0
    impact_factor.loc[impact_factor.groupby(['Unify venue'], sort=False)['Year'].diff(1) == 1, 'IF last year'] = \
        impact_factor.groupby(['Unify venue'], sort=False)['IF'].shift(1)
    eval_cols(impact_factor, ['Unify venue', 'Year', 'Cit count back 2 years', 'Paper count', 'Pub 2 years back', 'IF', 'IF last year'])
    print('\t' + '\t' + 'Impact factor time: ' + str(time.time() - base_time)); base_time = time.time()

    # H INDEX upto last year: redo.
    # 1st route:
    detail_cit_count = active_paper.loc[:, ['Unify venue', 'Paper ID']].merge(active_paper_paper, left_on='Paper ID', right_on='Paper ref ID')  # may smaller than author simple if some papers have no citation
    eval_cols(detail_cit_count, ['Unify venue', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID'])  # paper id x == paper ref id, paper y cites paper x
    detail_cit_count = detail_cit_count.merge(paper.loc[:, ['Paper ID', 'Year']], left_on='Paper ID'+'_y', right_on='Paper ID')  # get year of citing paper
    eval_cols(detail_cit_count, ['Unify venue', 'Paper ID'+'_x', 'Paper ID'+'_y', 'Paper ref ID', 'Paper ID', 'Year'])  # paper id y == paper id: citing paper
    # Count citation of each paper received in each year (note that papers are repeated for all coauthors).
    detail_cit_count = detail_cit_count.groupby(['Unify venue', 'Paper ID'+'_x', 'Year'], sort=False).size().reset_index(name='Cit count')
    detail_cit_count = detail_cit_count.rename(columns={'Paper ID'+'_x': 'Paper ID'})
    eval_cols(detail_cit_count, ['Unify venue', 'Paper ID', 'Year', 'Cit count'])
    # 2nd route:
    detail_year = active_paper.loc[:, ['Unify venue', 'Paper ID']] \
        .merge(active_paper.loc[:, ['Unify venue', 'Year']].drop_duplicates(), on=['Unify venue'])  # only repeat |num year| times: should not run out of ram
    eval_cols(detail_year, ['Unify venue', 'Paper ID', 'Year'])
    # collapse 2 routes:
    detail_cit_count = detail_cit_count.merge(detail_year, on=['Unify venue', 'Paper ID', 'Year'], how='outer')
    eval_cols(detail_cit_count, ['Unify venue', 'Paper ID', 'Year', 'Cit count'])
    detail_cit_count = detail_cit_count.fillna(0)
    # Sum citation count of EACH PAPER upto each CITING AND PUB year.
    detail_cit_count.loc[:, 'Cum cit count'] = detail_cit_count.sort_values(['Unify venue', 'Paper ID', 'Year']).groupby(['Unify venue', 'Paper ID'], sort=False)['Cit count'].cumsum()
    detail_cit_count.loc[:, 'Cum cit count upto last year'] = detail_cit_count.loc[:, 'Cum cit count'].values - detail_cit_count.loc[:, 'Cit count'].values
    eval_cols(detail_cit_count, ['Unify venue', 'Paper ID', 'Year', 'Cit count', 'Cum cit count', 'Cum cit count upto last year'])
    # Filter h5:
    detail_cit_count = detail_cit_count.merge(active_paper.loc[:, ['Paper ID', 'Year']], on='Paper ID')  # get year of cited paper, no need to outer, only care about papers that have citation
    eval_cols(detail_cit_count, ['Unify venue', 'Paper ID', 'Year'+'_x', 'Cit count', 'Cum cit count', 'Cum cit count upto last year', 'Year'+'_y'])
    detail_cit_count = detail_cit_count.rename(columns={'Year'+'_x': 'Year', 'Year'+'_y': 'Pub year'})
    h_window = 5
    detail_cit_count5 = detail_cit_count.loc[detail_cit_count.loc[:, 'Year'].values - detail_cit_count.loc[:, 'Pub year'].values <= h_window, :]  # publish year must be not too old with regards to the year to compute h index. No need to check lower bound, because already compute cum citcount upto last year.
    # Compute h index: alg: groupby author-year, sort by cum citcount desc (paper asc for deterministic), cumcount rank, select top h paper, group by author-year count paper as h index.
    detail_cit_count.loc[:, 'Rank'] = detail_cit_count.sort_values(['Unify venue', 'Year', 'Cum cit count upto last year', 'Paper ID'], ascending=[True, True, False, True]) \
        .groupby(['Unify venue', 'Year'], sort=False).cumcount() + 1
    detail_cit_count = detail_cit_count.loc[detail_cit_count.loc[:, 'Rank'].values <= detail_cit_count.loc[:, 'Cum cit count upto last year'].values, :]
    h_index = detail_cit_count.groupby(['Unify venue', 'Year'], sort=False).size().reset_index(name='H index')
    # Compute h5 index:
    detail_cit_count5.loc[:, 'Rank'] = detail_cit_count5.sort_values(['Unify venue', 'Year', 'Cum cit count upto last year', 'Paper ID'], ascending=[True, True, False, True]) \
        .groupby(['Unify venue', 'Year'], sort=False).cumcount() + 1
    detail_cit_count5 = detail_cit_count5.loc[detail_cit_count5.loc[:, 'Rank'].values <= detail_cit_count5.loc[:, 'Cum cit count upto last year'].values, :]
    h5_index = detail_cit_count5.groupby(['Unify venue', 'Year'], sort=False).size().reset_index(name='H5 index')
    print('\t' + '\t' + 'H index time: ' + str(time.time() - base_time)); base_time = time.time()

    # Merge:
    venue_simple = venue_simple \
        .merge(venue_runlength.loc[:, ['Unify venue', 'Year', 'Run len']], on=['Unify venue', 'Year'], how='left') \
        .merge(paper_count.loc[:, ['Unify venue', 'Year', 'Cum paper count upto last year', 'Paper count last year']], on=['Unify venue', 'Year'], how='left') \
        .merge(cit_count.loc[:, ['Unify venue', 'Year', 'Cum cit count upto last year', 'Cit count last year']], on=['Unify venue', 'Year'], how='left')\
        .merge(h_index.loc[:, ['Unify venue', 'Year', 'H index']], on=['Unify venue', 'Year'], how='left') \
        .merge(h5_index.loc[:, ['Unify venue', 'Year', 'H5 index']], on=['Unify venue', 'Year'], how='left') \
        .merge(impact_factor.loc[:, ['Unify venue', 'Year', 'IF last year']], on=['Unify venue', 'Year'], how='left')
    venue_simple = venue_simple.fillna(0)
    venue_simple.to_csv(os.path.join(root_path, args.save_dir, 'VENUE_SIMPLE' + '.txt'), sep=' ', header=True, index=False)

    venue_simple = None
    venue_runlength = None
    paper_count = None
    cit_count = None
    impact_factor = None
    h_index = None
    h5_index = None
    detail_cit_count = None
    detail_cit_count5 = None
    detail_year = None
    first_year = None
    print('\t' + 'Venue simple: Save time: ' + str(time.time() - base_time)); base_time = time.time()


    print('COMPUTE SIMPLE FEATURES: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


def read_simple_x(paper_align_filename, paper_simple_filename=None, author_simple_filename=None, venue_simple_filename=None,
                  pooling=['avg', 'max'], start_test_year=1996, end_test_year=2000, period=5, trim_combine=[1, 1, 0], fillna='0'):
    """Read simple features input.
    Align by paper id in paper align.

    Notice: all features are filtered and aligned based on paper_align.
    -> Need to guarantee that paper has author and venue, then check alignment and fill na value to empty value.

    Note:
        - After combining, separate to train and test. (remember to discard 5 years before test year, e.g., 1991-1995.)

    Modularize.

    :param paper_align_filename:
    :param paper_simple_filename:
    :param author_simple_filename:
    :param venue_simple_filename:
    :param pooling: [['avg', 'max'], 'sum'].
    :param start_test_year:
    :param end_test_year:
    :param period:
    :param trim_combine: [1, 1, 0] for [trim by paper, trim by author, no trim by venue], 1 means trim papers when a combined embeddings lacking that paper id, 0 means keep all papers in paper_align, fillna in embedding by 0 or mean().
    :param fillna: how to fillna when not trim_combine: 'avg' or a value to fill in. Default: average.
    :return: matrixes train_x and test_x (pandas df).
    """

    print('READ SIMPLE X.')
    start_time = time.time()  # in second.

    # Read paper align.
    base_time = time.time()
    names_paper_align = ['Year', 'Paper ID', 'Author ID', 'Venue ID']
    dtype_paper_align = {'Year': int, 'Paper ID': str, 'Author ID': str, 'Venue ID': str}
    paper_align = pd.read_csv(paper_align_filename, delimiter=' ', header=None, names=names_paper_align, dtype=dtype_paper_align, engine='c')
    paper_align = paper_align.loc[(paper_align.loc[:, 'Year'] < start_test_year - period)
                                  | ((paper_align.loc[:, 'Year'] >= start_test_year) & (paper_align.loc[:, 'Year'] <= end_test_year)), :]
    print('\tRead paper align: {}'.format(str(time.time() - base_time))); base_time = time.time()

    x = paper_align.loc[:, ['Paper ID', 'Year']].drop_duplicates()
    print('\tInit x: {}'.format(str(time.time() - base_time))); base_time = time.time()

    # Read each embeddings and combine to x.
    # Modularizing: for each embedding file, read, compute, combine.
    # Data structure:
    #   first row: header
    #   other rows: str paperid [simple features]
    if paper_simple_filename is not None:
        # Read.
        paper = pd.read_csv(paper_simple_filename, delimiter=' ', header=0, engine='c')
        eval_cols(paper, ['Paper ID', 'Author count', 'Ref count', 'Title len', 'Month'])
        print('\tRead paper: {}'.format(str(time.time() - base_time))); base_time = time.time()
        # No further computation, just use.
        # Combine.
        if trim_combine[0]:
            x = x.merge(paper, on='Paper ID')
        else:
            x = x.merge(paper, on='Paper ID', how='left')
        print('\tCombine paper: {}'.format(str(time.time() - base_time))); base_time = time.time()

    if author_simple_filename is not None:
        # Read.
        author = pd.read_csv(author_simple_filename, delimiter=' ', header=0, engine='c')
        eval_cols(author, ['Author ID', 'Year',
                           'Cum coauthor count upto last year', 'Career len',
                           'Cum paper count upto last year', 'Paper count last year',
                           'Cum cit count upto last year', 'Cit count last year',
                           'H index', 'H5 index', 'IF last year'])
        author = author.rename(columns={'Cum paper count upto last year': 'Cum paper count upto last year' + ' author',
                                        'Paper count last year': 'Paper count last year' + ' author',
                                        'Cum cit count upto last year': 'Cum cit count upto last year' + ' author',
                                        'Cit count last year': 'Cit count last year' + ' author',
                                        'H index': 'H index' + ' author',
                                        'H5 index': 'H5 index' + ' author',
                                        'IF last year': 'IF last year' + ' author'})
        print('\tRead author: {}'.format(str(time.time() - base_time))); base_time = time.time()
        # Compute. Both avg and max.
        author = paper_align.loc[:, ['Paper ID', 'Author ID', 'Year']].merge(author, on=['Author ID', 'Year']).drop(['Author ID', 'Year'], axis=1)  # (no right join) Discard all features with no matching info in paper_align, because do not know how to align them. (no left join) Discard all paper_align with no matching in embeddings, because this is just intermediate step, just keep what are in embeddings. So inner join.
        print('\tMerge author: {}'.format(str(time.time() - base_time))); base_time = time.time()
        if 'avg' in pooling and 'max' in pooling:  # quick prototype here, do both mean and max, will make option more flexible later.
            author = author.groupby('Paper ID').agg(['mean', 'max'])
            author.columns = [' '.join(leveledlabels) for leveledlabels in author.columns]
            author = author.reset_index()
        print('\tCompute mean, max author: {}'.format(str(time.time() - base_time))); base_time = time.time()
        # Combine.
        if trim_combine[1]:
            x = x.merge(author, on='Paper ID')
        else:
            x = x.merge(author, on='Paper ID', how='left')
        print('\tCombine author: {}'.format(str(time.time() - base_time))); base_time = time.time()

    if venue_simple_filename is not None:
        # Read.
        venue = pd.read_csv(venue_simple_filename, delimiter=' ', header=0, engine='c')
        eval_cols(venue, ['Unify venue', 'Year',
                          'Run len',
                          'Cum paper count upto last year', 'Paper count last year',
                          'Cum cit count upto last year', 'Cit count last year',
                          'H index', 'H5 index', 'IF last year'])
        venue = venue.rename(columns={'Unify venue': 'Venue ID',
                                      'Cum paper count upto last year': 'Cum paper count upto last year' + ' venue',
                                      'Paper count last year': 'Paper count last year' + ' venue',
                                      'Cum cit count upto last year': 'Cum cit count upto last year' + ' venue',
                                      'Cit count last year': 'Cit count last year' + ' venue',
                                      'H index': 'H index' + ' venue',
                                      'H5 index': 'H5 index' + ' venue',
                                      'IF last year': 'IF last year' + ' venue'})
        print('\tRead venue: {}'.format(str(time.time() - base_time))); base_time = time.time()
        venue.loc[:, 'isConf'] = [1 if pd.notnull(vid) and vid.startswith('C') else 0 for vid in venue.loc[:, 'Venue ID'].values]  # add dummy feature
        venue.loc[:, 'isJour'] = [1 if pd.notnull(vid) and vid.startswith('J') else 0 for vid in venue.loc[:, 'Venue ID'].values]  # add dummy feature
        print('\tAdd venue type: {}'.format(str(time.time() - base_time))); base_time = time.time()
        # Compute.
        venue = paper_align.loc[:, ['Paper ID', 'Venue ID', 'Year']].drop_duplicates().merge(venue, on=['Venue ID', 'Year']).drop(['Venue ID', 'Year'], axis=1)
        print('\tMerge venue: {}'.format(str(time.time() - base_time))); base_time = time.time()
        # Combine.
        if trim_combine[2]:
            x = x.merge(venue, on='Paper ID')
        else:
            x = x.merge(venue, on='Paper ID', how='left')
            x.loc[:, ['isConf', 'isJour']] = x.loc[:, ['isConf', 'isJour']].fillna(0).astype(int)  # there are many papers without venue info, if do not trim, venue type is unknown, it's also a feature showing that venue info is missing or not
        print('\tCombine venue: {}'.format(str(time.time() - base_time))); base_time = time.time()

    if fillna == 'avg':
        x = x.fillna(x.mean())
        x = x.fillna(0.0)  # fill 0 to columns containing all NA values.
    else:
        x = x.fillna(float(fillna))
    print('\tFill NA: {}'.format(str(time.time() - base_time))); base_time = time.time()  # Fill NA mean() is too slow.

    x = x.sort_values('Paper ID')  # Make result reproducible.
    print('\tSort x: {}'.format(str(time.time() - base_time))); base_time = time.time()

    x.columns = [c.replace(" ", "_") for c in x.columns]

    print('READ SIMPLE X: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    return x.loc[x.loc[:, 'Year'] < start_test_year - period, :], \
        x.loc[(x.loc[:, 'Year'] >= start_test_year) & (x.loc[:, 'Year'] <= end_test_year), :]


def read_highimpact_y(venue_paper_rank_filename, min_paper_count=10):
    """Read ground truth output of highimpact prediction problem.
    Note that some papers in x do not have full information in y, we will filter out these papers later.

    * Expected output:
        - each venue -> list of papers with corresponding rankings. Note that list of papers in known, only ranking is unknown.
        - format: paper-ranking pair, but need a way to group paper by venue -> pandas df, columns: venueid-year-paperid-ranking(-citcount is ok)
    * Evaluation:
        - just do precision/recall/f1 at 5, ndcg at 5, mrr at 5 average for all venues.
        - evaluation processing after prediction.

    :return: y (structural pandas df: ['Unify venue', 'Year', 'Paper ID', 'Size', 'Rank', 'Paper count'])
    """

    print('READ HIGH IMPACT Y.')
    start_time = time.time()  # in second.

    y = pd.read_csv(venue_paper_rank_filename, delimiter=' ', header=0, engine='c')
    eval_cols(y, ['Unify venue', 'Year', 'Paper ID', 'Size', 'Rank'])

    # Filter noisy venue:
    venue_paper_count = y.groupby(['Unify venue', 'Year'], sort=False).size().reset_index(name='Paper count')
    y = y.merge(venue_paper_count.loc[venue_paper_count.loc[:, 'Paper count'].values >= min_paper_count, ['Unify venue', 'Year']], on=['Unify venue', 'Year'])
    eval_cols(y, ['Unify venue', 'Year', 'Paper ID', 'Size', 'Rank'])

    y.columns = [c.replace(" ", "_") for c in y.columns]

    print('READ HIGH IMPACT Y: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    return y


def parse_args():
    """Parses the arguments.
    """

    global debug  # claim to use global var.

    parser = argparse.ArgumentParser(description="Prepare data for high impact prediction.")

    parser.add_argument('--root-path', default=None,
                        help="Root folder path. Default None.")

    parser.add_argument('--save-dir', default='HighImpact',
                        help="Save folder. Default 'HighImpact'.")

    parser.add_argument('--temp-dir', default='temp',
                        help='Temp folder. Default "temp".')

    parser.add_argument('--debug-server', dest='debug_server', action='store_true',
                        help='Turn on debug mode on server. Default: off.')
    parser.set_defaults(debug_server=False)

    parser.add_argument('--start-test-year', type=int, default=1996,
                        help='The start test year. Default: 1996.')
    parser.add_argument('--end-test-year', type=int, default=2000,
                        help='The end test year. Default: 2000.')
    parser.add_argument('--period', type=int, default=5,
                        help='Period after publication. Default: 5.')
    parser.add_argument('--min-year', type=int, default=1970,
                        help='The min year. Default: 1970.')

    parser.add_argument('--no-save-align', dest='save_align', action='store_false',
                        help='Do not save paper align. Default: save.')
    parser.set_defaults(save_align=True)

    parser.add_argument('--no-save-citcount', dest='save_citcount', action='store_false',
                        help='Do not save cit count sequence. Default: save.')
    parser.set_defaults(save_citcount=True)

    parser.add_argument('--no-save-venue-paper-rank', dest='save_venue_paper_rank', action='store_false',
                        help='Do not save venue paper rank in the period. Default: save.')
    parser.set_defaults(save_venue_paper_rank=True)

    parser.add_argument('--config', nargs='*', default=[],
                        help='Config. Default: [].')

    args = parser.parse_args()

    # Post-process args value.
    if args.root_path is None:
        root_path_local = "./data/MAGSample"
        root_path_server = "~/Data/MAG/Unzip"
        if os.path.isdir(root_path_local):
            args.root_path = root_path_local
            debug = True
        else:
            args.root_path = root_path_server
    if not os.path.isdir(os.path.join(args.root_path, args.save_dir)):
        os.makedirs(os.path.join(args.root_path, args.save_dir))
    if not os.path.isdir(os.path.join(args.root_path, args.temp_dir)):
        os.makedirs(os.path.join(args.root_path, args.temp_dir))

    # DEBUG.
    if debug:
        pass
    if args.debug_server and 'unittest' in args.config:
        args.save_align = False
        args.save_citcount = False
        args.save_venue_paper_rank = False

    # Finally return.
    return args


if __name__ == '__main__':
    debug = False
    main(parse_args())
