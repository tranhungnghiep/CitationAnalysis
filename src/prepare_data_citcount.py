"""
Prepare data for citation count prediction.
Data source: Microsoft Academic Graph (MAG)

Outline:
* Network:
    - Read MAG csv files.
    - Filter paper by domain (computer science).
    - Add UNIFYVENUE column.
    - Filter 'noisy' paper.
    - Extract paper align.
    - Extract cit count sequence.
    - For each test year.
        For each entity:
            paper
            author
            venue
                Select citnet.
                Select authorship-based net.
                Select submission-based net.
                (for both train and test)
    - Save all to files.
* Compute embedding: using node2vec.
* Read and align embeddings.
"""


import os
import time
import argparse

import pandas as pd
import numpy as np
np.random.seed(7)
import random
random.seed(7)

import multiprocessing
import joblib


def main(args):
    print('START.')
    start_time_main = time.time()  # in second.


    build_network(args)


    print('FINISH.')
    stop_time_main = time.time()
    print('Time (s): ' + str(stop_time_main-start_time_main))

    return 0


def build_network(args):
    """Port from SQl to pandas code.
    Build 7 networks and save to files.
    """

    global debug

    # Working folder.
    root_path = args.root_path

    start_test_year = args.start_test_year
    end_test_year = args.end_test_year
    min_year_step = args.min_year_step
    max_year_step = args.max_year_step
    min_year = args.min_year
    max_year = end_test_year + max_year_step  # 2005


    # 1. Read csv files.
    print('READ FILES.')
    start_time = time.time()  # in second.

    paper_filename = 'Papers.txt'
    names_paper = ['Paper ID', 'Orig title', 'Norm title', 'Year', 'Date', 'DOI', 'Venue name', 'Norm venue name', 'Journal ID', 'Conference ID', 'Rank']
    usecols_paper = ['Paper ID', 'Orig title', 'Year', 'Journal ID', 'Conference ID']
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

    # Performance pandas read_csv:
    # Important option:
    #   dtype=str, then as_type(int) year, to avoid infering dtype and problems like mixed dtype, but pandas inferencer is pretty smart and not many columns so this does not affect much.
    #   engine='c' is default already.
    # Large ram:
    #   memory_map=True to swap file to ram before processing, seems not helpful need more ram and even slightly slower.
    #   low_memory=False to process file at once, not really related or helpful to performance.
    # Other:
    #   na_filter=False, should not use, because data contain NaN value.
    dtype_paper = {'Paper ID': str, 'Orig title': str, 'Norm title': str, 'Year': int, 'Date': str, 'DOI': str, 'Venue name': str, 'Norm venue name': str, 'Journal ID': str, 'Conference ID': str, 'Rank': int}

    # Sequential reading files:
    # base_time = time.time()
    # paper = pd.read_csv(paper_file, delimiter='\t', header=None, skiprows=0, names=names_paper, usecols=usecols_paper, dtype=dtype_paper, engine='c')
    # print('\t' + 'Read paper: ' + str(time.time() - base_time)); base_time = time.time()
    # # paper.loc[:, 'Year'] = paper.loc[:, 'Year'].values.astype(int)  # 102s.
    # # print('\t' + 'Year as type int: ' + str(time.time() - base_time)); base_time = time.time()
    # paper_paper = pd.read_csv(paper_paper_file, delimiter='\t', header=None, skiprows=0, names=names_paper_paper, usecols=usecols_paper_paper, dtype=str, engine='c')
    # print('\t' + 'Read paper paper: ' + str(time.time() - base_time)); base_time = time.time()
    # paper_author = pd.read_csv(paper_author_file, delimiter='\t', header=None, skiprows=0, names=names_paper_author, usecols=usecols_paper_author, dtype=str, engine='c')
    # print('\t' + 'Read paper author: ' + str(time.time() - base_time)); base_time = time.time()
    # paper_keyword = pd.read_csv(paper_keyword_file, delimiter='\t', header=None, skiprows=0, names=names_paper_keyword, usecols=usecols_paper_keyword, dtype=str, engine='c')
    # print('\t' + 'Read paper keyword: ' + str(time.time() - base_time)); base_time = time.time()
    # domain = pd.read_csv(domain_file, delimiter='\t', header=None, skiprows=0, names=names_domain, usecols=usecols_domain, dtype=str, engine='c')
    # print('\t' + 'Read domain: ' + str(time.time() - base_time)); base_time = time.time()
    # domain_level = pd.read_csv(domain_level_file, delimiter='\t', header=None, skiprows=0, names=names_domain_level, usecols=usecols_domain_level, dtype=str, engine='c')
    # print('\t' + 'Read domain level: ' + str(time.time() - base_time)); base_time = time.time()

    # Parallel reading files: IO heavy should use threads instead of processes. Process fails: much longer than sequential reading and consuming much more ram.
    import multiprocessing.pool
    pool = multiprocessing.pool.ThreadPool(6)  # Use 6 threads for 6 files.
    paper = pool.apply_async(pd.read_csv, args=(paper_file, ), kwds={'delimiter': '\t', 'header': None, 'skiprows': 0, 'names': names_paper, 'usecols': usecols_paper, 'dtype': dtype_paper, 'engine': 'c'})
    paper_paper = pool.apply_async(pd.read_csv, args=(paper_paper_file, ), kwds={'delimiter': '\t', 'header': None, 'skiprows': 0, 'names': names_paper_paper, 'usecols': usecols_paper_paper, 'dtype': str, 'engine': 'c'})
    paper_author = pool.apply_async(pd.read_csv, args=(paper_author_file, ), kwds={'delimiter': '\t', 'header': None, 'skiprows': 0, 'names': names_paper_author, 'usecols': usecols_paper_author, 'dtype': str, 'engine': 'c'})
    paper_keyword = pool.apply_async(pd.read_csv, args=(paper_keyword_file, ), kwds={'delimiter': '\t', 'header': None, 'skiprows': 0, 'names': names_paper_keyword, 'usecols': usecols_paper_keyword, 'dtype': str, 'engine': 'c'})
    domain = pool.apply_async(pd.read_csv, args=(domain_file, ), kwds={'delimiter': '\t', 'header': None, 'skiprows': 0, 'names': names_domain, 'usecols': usecols_domain, 'dtype': str, 'engine': 'c'})
    domain_level = pool.apply_async(pd.read_csv, args=(domain_level_file, ), kwds={'delimiter': '\t', 'header': None, 'skiprows': 0, 'names': names_domain_level, 'usecols': usecols_domain_level, 'dtype': str, 'engine': 'c'})
    pool.close()  # Close pool after all processes finished.
    pool.join()  # Wait untill pool close.
    paper = paper.get()
    paper_paper = paper_paper.get()
    paper_author = paper_author.get()
    paper_keyword = paper_keyword.get()
    domain = domain.get()
    domain_level = domain_level.get()

    print('READ FILES: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


    # 2. Filter by domain 'Computer Science'.
    print('FILTER BY DOMAIN.')
    start_time = time.time()  # in second.

    if debug:
        # First, for sanity check, save all domain level 0.
        # Use .loc[:, []] return a dataframe instead of series. Should use this most of the time (like before drop_duplicates, sort_values...). But do not use this when using the result for masking.
        # domain_level0 = domain.merge(domain_level.loc[domain_level.loc[:, 'Parent level'].values == 'L0', ['Parent ID']].drop_duplicates(), left_on='Domain ID', right_on='Parent ID') \
        #     .loc[:, ['Domain ID', 'Domain name']].sort_values('Domain name').reset_index(drop=True)
        domain_level0 = domain.loc[domain.loc[:, 'Domain ID'].isin(domain_level.loc[domain_level.loc[:, 'Parent level'].values == 'L0', 'Parent ID'].drop_duplicates().values), :] \
            .sort_values('Domain name').reset_index(drop=True)  # isin() is faster than merge: 17ms vs 44ms.
        domain_level0.to_csv(os.path.join(root_path, args.save_dir, 'domain_level0.csv'))

    # Confirmed, domain level 0 'Computer Science', Domain ID = 0271BC14.
    domain_cs = pd.DataFrame(['0271BC14'], columns=['Domain ID'])
    while True:
        # domain_cs_temp = domain_cs.merge(domain_level, left_on='Domain ID', right_on='Parent ID').loc[:, ['Parent ID', 'Child ID']]
        domain_cs_temp = domain_level.loc[domain_level.loc[:, 'Parent ID'].isin(domain_cs.loc[:, 'Domain ID'].values), ['Parent ID', 'Child ID']]  # isin() is faster than merge: 14ms vs 25ms.
        domain_cs_temp = pd.DataFrame(np.concatenate((domain_cs_temp.loc[:, ['Parent ID']].values, domain_cs_temp.loc[:, ['Child ID']].values)), columns=['Domain ID']).drop_duplicates()
        print('\t' + 'Check domain level complete: ' + str(domain_cs.shape == domain_cs_temp.shape and (domain_cs.values == domain_cs_temp.values).all().all()))
        if domain_cs.shape == domain_cs_temp.shape and (domain_cs.values == domain_cs_temp.values).all().all():  # Note that comparison with None or NaN is always False.
            break
        else:
            domain_cs = domain_cs_temp

    # Then filter paper in CS.
    # paper_id = paper_keyword.merge(domain_cs, left_on='Domain ID', right_on='Domain ID').loc[:, ['Paper ID']].drop_duplicates()
    # paper = paper.merge(paper_id, left_on='Paper ID', right_on='Paper ID')  # Result contains only 1 Paper ID column, no need to use .loc[]. Moreover, using .loc with mixed correct and wrong column names do not raise error, but return wrong data in a new null column. Should always check carefully, better yet write a helper function that check that all columns are correct.
    # eval_cols(paper, ['Paper ID', 'Orig title', 'Year', 'Journal ID', 'Conference ID'])  # CHECKPOINT.
    paper_id = paper_keyword.loc[paper_keyword.loc[:, 'Domain ID'].isin(domain_cs.loc[:, 'Domain ID'].values), ['Paper ID']].drop_duplicates()  # isin: 17ms, merge: 19 ms.
    paper = paper.loc[paper.loc[:, 'Paper ID'].isin(paper_id.loc[:, 'Paper ID'].values), :]  # isin: 3ms, merge: 18ms.

    # Clean up.
    domain = None
    domain_level = None
    paper_keyword = None

    print('FILTER BY DOMAIN: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


    # 3. Add UNIFY VENUE column.
    print('UNIFY VENUE.')
    start_time = time.time()  # in second.

    # Note that venue id is not int as in MAS, but length-8 string.
    # For each paper, 1 or both venue id are NA.
    # solution 1: fillna then concat string and put it in unify_venue column. to invert: split string.
    # solution 2: update J or C to venue id and put it in unify_venue column. to invert: check prefix. use this solution.
    # paper.loc[:, 'Unify venue'] = paper.apply(unify_venue, axis=1)
    # Use numpy: FASTER: x2.
    # paper.loc[:, 'Unify venue'] = paper.apply(unify_venue_np, axis=1, raw=True, args=(paper.columns.get_loc("Journal ID"), paper.columns.get_loc("Conference ID")))
    # Use numpy + list comprehension: MUCH MUCH FASTER: x50.
    j = paper.columns.get_loc("Journal ID")
    c = paper.columns.get_loc("Conference ID")
    paper.loc[:, 'Unify venue'] = [unify_venue_np(row, j, c) for row in paper.values]
    # paper.loc[:, 'Unify venue'] = joblib.Parallel(n_jobs=args.parallel_workers)(joblib.delayed(function=unify_venue_np)(row, j, c) for row in paper.values)  # Parallelize. Had better use Pool.starmap because Pool is faster than joblib, but it's in Python 3. Somethings wrong, parallel is very slow, it takes very long before starting multiple processes. After that, there are too much small processes, the overhead is too high and not efficient.
    paper = paper.loc[:, ['Paper ID', 'Orig title', 'Year', 'Unify venue']]

    print('UNIFY VENUE: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


    # 4. Filter 'noisy' paper.
    # ONLY USE PAPERS FROM 1970 TO 2010 (TRIMMED TO 2005), WITH NOT NULL TITLE, VENUE, AUTHOR. (KEYWORD, DOMAIN filtered above).
    print('FILTER NOISY PAPER AND BUILD PAPER ALIGN.')
    start_time = time.time()  # in second.

    # Convert to numpy array before comparing to produce no-label masking arrays, use bitwise '&' to merge 2 logical array. Note that comparison to None or NaN will result in False.
    paper = paper.loc[min_year <= paper.loc[:, 'Year'].values, :]
    paper = paper.loc[paper.loc[:, 'Year'].values <= max_year, :]  # Separate loc is faster than bitwise &.
    paper = paper.loc[pd.notnull(paper.loc[:, 'Orig title'].values), ['Paper ID', 'Year', 'Unify venue']]
    paper = paper.loc[pd.notnull(paper.loc[:, 'Unify venue'].values), :]
    paper_align = paper.merge(paper_author, left_on='Paper ID', right_on='Paper ID')  # This filters both paper and author: paper must have author, author's paper must be clean. Must use merge, cannot use isin, because need data from both df.
    eval_cols(paper_align, ['Paper ID', 'Year', 'Unify venue', 'Author ID'])  # CHECKPOINT.
    paper_align = paper_align.drop_duplicates()  # Just to make sure.

    # Clean up.
    paper = None
    paper_author = None

    print('FILTER NOISY PAPER AND BUILD PAPER ALIGN: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    # Trim paper_paper by filtered papers (1970-2005, Computer Science, no noise).
    # Only use clean CS citation, because citation network and citation count only has meaning in this case.
    #   => For each citation, both cit and ref must be in filtered papers.
    print('TRIM PAPER_PAPER BY PAPER_ALIGN.')
    start_time = time.time()  # in second.

    paper_paper = paper_paper.loc[paper_paper.loc[:, 'Paper ID'].isin(paper_align.loc[:, 'Paper ID'].values), :]
    paper_paper = paper_paper.loc[paper_paper.loc[:, 'Paper ref ID'].isin(paper_align.loc[:, 'Paper ID'].values), :]  # isin: 33ms, merge: 55ms. Separate loc is faster than bitwise. Only need to use bitwise |, for bitwise & better use separate loc.
    paper_paper = paper_paper.drop_duplicates()  # Just to make sure.

    print('TRIM PAPER_PAPER BY PAPER_ALIGN: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    # Trim paper_align by paper_paper.
    # Notice the induction relationship here: some papers are seeds, they cite no other papers, other papers cite them.
    #   => cannot filter out papers with no ref.
    #   => but can filter out isolated papers: papers with no cit and no ref.
    print('TRIM PAPER_ALIGN BY PAPER_PAPER.')
    start_time = time.time()  # in second.

    paper_align = paper_align.loc[paper_align.loc[:, 'Paper ID'].isin(paper_paper.loc[:, ['Paper ID', 'Paper ref ID']].values.ravel()), :]  # isin(series.values) is bit faster than isin(series). large.isin(small) is a bit faster than small.isin(large), but the resulting boolean mask is very large (same size as the large series, use lots of ram). Order is important, it has the meaning, hereby trimming paper_align after trimming paper_paper, meaning prioritizing cutting all noise and scaling down.

    print('TRIM PAPER_ALIGN BY PAPER_PAPER: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time - start_time))

    print('After cleaning, get back to unique paper.')
    paper = paper_align.loc[:, ['Paper ID', 'Orig title', 'Year', 'Date', 'Unify venue']].drop_duplicates  # This is the unique paper, do not need to drop duplicate from paper align all the time.

    print('Active data are only upto year ' + str(end_test_year))
    active_paper_align = paper_align.loc[paper_align.loc[:, 'Year'].values <= end_test_year, :]
    active_paper_paper = paper_paper.loc[paper_paper.loc[:, 'Paper ref ID'].isin(active_paper_align.loc[:, 'Paper ID'].values), :]
    active_paper = paper.loc[paper.loc[:, 'Year'].values <= end_test_year, :]

    # e.g.:
    # p_al: [p1, p2, p3, p5]
    # p_p: [p1->p2, p1->p4, p4->p5, p4->p1, p3->p6, p7->p8]
    # p_al_trim_or_before = [p1, p2, p3, p5] => keep p3 and p5 because they appear in cit or ref.
    # p_p_trim_and = [p1->p2] => both paper cit and ref have to be in paper_align. => only use citation in no noisy CS papers. Citcount and citation network (for author/venue) only has meaning in this case.
    # p_al_trim_or_after = [p1, p2] => only keep p1 and p2. => anyway, only p1, p2 are used in read x.
    #   p_p_trim_or = [p1->p2, p1->p4, p4->p5, p4->p1, p3->p6] => either paper cit and ref have to be in paper_align. => noisy, not used in citcount and citnet, only inflate paper citnet.
    #   p_al_trim_or_after_or = [p1, p2, p3, p5] => keep p3 and p5. => alright.
    #   p_p_trim_left = [p1->p2, p1->p4, p3->p6] => either paper cit and ref have to be in paper_align. => noisy, not used in citcount and citnet, only inflate paper citnet.
    #   p_al_trim_or_after_left = [p1, p2, p3] => keep p3 and p5.
    # Many options, should prioritize scaling down and removing noise.
    #   => p_p trim and.
    #   => p_al trim or after.
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


    # 6. Extract cit count sequence. And save.
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


    # 7. Select network: for each test year, select both train and test for paper/author/venue.
    print('SAVE NETWORKS.')
    start_time = time.time()  # in second.

    if args.ignore_citation:
        # Clean up.
        paper_paper = None

    for test_year in range(start_test_year, end_test_year + 1):  # [1996, ..., 2000]
        base_time = time.time()
        print('SAVE NETWORK FOR TEST YEAR: ' + str(test_year))

        if 'Fast6' in args.config:
            # Incremental building networks for later years.
            # Todo: if test_year == 1997-2000: reuse old network (var outside for-loop), only build subnetwork for paper_align in current test_year, then combine with old network.
            pass

        # Trim papers by the current test year.
        global paper_align_i
        paper_align_i = paper_align.loc[paper_align.loc[:, 'Year'].values <= test_year, :]
        print('\t' + 'Trim papers by year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()
        if not args.ignore_citation:
            # Trim paper_paper by the current test year.
            paper_paper_i = paper_paper.loc[paper_paper.loc[:, 'Paper ID'].isin(paper_align_i.loc[:, 'Paper ID'].values), :]  # Paper ref is clean, as trimmed above, so no need to filter now as they are in the past.
            print('\t' + 'Trim paper_paper by year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()

        if 1 in args.network:
            # 1. Paper citnet.
            paper_citnet = paper_paper_i
            paper_citnet = paper_citnet.drop_duplicates()  # Just to make sure.
            paper_citnet.to_csv(os.path.join(root_path, args.save_dir, 'PAPER_CITATION_NETWORK_' + str(test_year) + '.txt'), sep=' ', header=False, index=False)
            print('\t' + '# Paper citnet nodes: ' + str(len(pd.unique(paper_citnet.loc[:, ['Paper ID', 'Paper ref ID']].values.ravel()))))  # CHECKPOINT.
            paper_citnet = None
            print('\t' + 'Paper citnet year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()
        if 2 in args.network:
            # 2. Author citnet.
            author_citnet = paper_paper_i \
                .merge(paper_align_i.loc[:, ['Paper ID', 'Author ID']], left_on='Paper ID', right_on='Paper ID') \
                .merge(paper_align_i.loc[:, ['Paper ID', 'Author ID']], left_on='Paper ref ID', right_on='Paper ID')  # Must use merge, cannot use isin, because need data from both df.
            eval_cols(author_citnet, ['Paper ID'+'_x', 'Paper ref ID', 'Author ID'+'_x', 'Paper ID'+'_y', 'Author ID'+'_y'])  # CHECKPOINT.
            author_citnet = author_citnet.groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')  # How many times author x cites author y.
            eval_cols(author_citnet, ['Author ID'+'_x', 'Author ID'+'_y', 'Size'])  # CHECKPOINT.
            # author_citnet = author_citnet.drop_duplicates()  # Just to make sure.
            for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                author_citnet.loc[author_citnet.loc[:, 'Size'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'AUTHOR_CITATION_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False)
            print('\t' + '# Author citnet nodes: ' + str(len(pd.unique(author_citnet.loc[:, ['Author ID'+'_x', 'Author ID'+'_y']].values.ravel()))))  # CHECKPOINT.
            author_citnet = None
            print('\t' + 'Author citnet year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()
        if 3 in args.network:
            # 3. Venue citnet.
            venue_citnet = paper_paper_i \
                .merge(paper_align_i.loc[:, ['Paper ID', 'Unify venue']].drop_duplicates(), left_on='Paper ID', right_on='Paper ID') \
                .merge(paper_align_i.loc[:, ['Paper ID', 'Unify venue']].drop_duplicates(), left_on='Paper ref ID', right_on='Paper ID')  # Must use merge, cannot use isin, because need data from both df.
            eval_cols(venue_citnet, ['Paper ID'+'_x', 'Paper ref ID', 'Unify venue'+'_x', 'Paper ID'+'_y', 'Unify venue'+'_y'])  # CHECKPOINT.
            venue_citnet = venue_citnet.groupby(['Unify venue'+'_x', 'Unify venue'+'_y'], sort=False).size().reset_index(name='Size')  # How many times venue x cites venue y.
            eval_cols(venue_citnet, ['Unify venue'+'_x', 'Unify venue'+'_y', 'Size'])  # CHECKPOINT.
            # venue_citnet = venue_citnet.drop_duplicates()  # Just to make sure.
            for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                venue_citnet.loc[venue_citnet.loc[:, 'Size'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'VENUE_CITATION_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False)
            print('\t' + '# Venue citnet nodes: ' + str(len(pd.unique(venue_citnet.loc[:, ['Unify venue'+'_x', 'Unify venue'+'_y']].values.ravel()))))  # CHECKPOINT.
            venue_citnet = None
            print('\t' + 'Venue citnet year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()

        if 4 in args.network:
            # 4. Share-author paper network.
            # Option 1: Fully merge.
            paper_sanet = paper_align_i.loc[:, ['Paper ID', 'Author ID']] \
                .merge(paper_align_i.loc[:, ['Paper ID', 'Author ID']], left_on='Author ID', right_on='Author ID')
            eval_cols(paper_sanet, ['Paper ID'+'_x', 'Author ID', 'Paper ID'+'_y'])  # CHECKPOINT.
            paper_sanet = paper_sanet.loc[paper_sanet.loc[:, 'Paper ID'+'_x'].values < paper_sanet.loc[:, 'Paper ID'+'_y'].values, :]  # Not allow loop edge and duplicate undirected edge.
            paper_sanet = paper_sanet.groupby(['Paper ID'+'_x', 'Paper ID'+'_y'], sort=False).size().reset_index(name='Size')  # How many matched authors between paper x and paper y.
            # Common.
            if not paper_sanet.empty:
                eval_cols(paper_sanet, ['Paper ID'+'_x', 'Paper ID'+'_y', 'Size'])  # CHECKPOINT.
                # paper_sanet = paper_sanet.drop_duplicates()  # Just to make sure.
                for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                    paper_sanet.loc[paper_sanet.loc[:, 'Size'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'PAPER_SHARE_AUTHOR_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False)
                print('\t' + '# Share-author paper net nodes: ' + str(len(pd.unique(paper_sanet.loc[:, ['Paper ID'+'_x', 'Paper ID'+'_y']].values.ravel()))))  # CHECKPOINT.
                paper_sanet = None
            print('\t' + 'Share-author paper net year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()

        if 5 in args.network or 6 in args.network:
            # 5. Share-paper author (co-author) network.
            # Option 1: Fully merge.
            author_spnet = paper_align_i.loc[:, ['Paper ID', 'Author ID']] \
                .merge(paper_align_i.loc[:, ['Paper ID', 'Author ID']], left_on='Paper ID', right_on='Paper ID')
            eval_cols(author_spnet, ['Paper ID', 'Author ID'+'_x', 'Author ID'+'_y'])  # CHECKPOINT.
            author_spnet = author_spnet.loc[author_spnet.loc[:, 'Author ID'+'_x'].values < author_spnet.loc[:, 'Author ID'+'_y'].values, :]  # Not allow loop edge and duplicate undirected edge.
            author_spnet = author_spnet.groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')  # How many matched papers between author x and author y.
            # Common.
            if not author_spnet.empty:
                eval_cols(author_spnet, ['Author ID'+'_x', 'Author ID'+'_y', 'Size'])  # CHECKPOINT.
                # author_spnet = author_spnet.drop_duplicates()  # Just to make sure.
                if 5 in args.network:
                    for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                        author_spnet.loc[author_spnet.loc[:, 'Size'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'AUTHOR_SHARE_PAPER_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False)
                    print('\t' + '# Share-paper author net nodes: ' + str(len(pd.unique(author_spnet.loc[:, ['Author ID'+'_x', 'Author ID'+'_y']].values.ravel()))))  # CHECKPOINT.
                if 'NoneCoAuthor' in args.config:
                    author_spnet = None  # Used in network 6.
            print('\t' + 'Share-paper author net year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()

        if 6 in args.network:
            # 6. Share-venue author network.
            # Note that share-venue author network includes share-paper author network, because coauthors are also counted.
            # This should be avoided, because it is noisy repeated info which may hide other useful infor, coauthor network already captures it,
            # and it is not the nature of this network: authors of same paper usually do not go together.
            if any(x.startswith('6.1') for x in args.config):
                # 6.1 Option 1: Fully merge. => out of memory: the merge result is too large.
                if '6.1V1' in args.config:
                    author_svnet = paper_align_i.loc[:, ['Unify venue', 'Year', 'Author ID']].drop_duplicates() \
                        .merge(paper_align_i.loc[:, ['Unify venue', 'Year', 'Author ID']].drop_duplicates(), left_on=['Unify venue', 'Year'], right_on=['Unify venue', 'Year'])
                    eval_cols(author_svnet, ['Unify venue', 'Year', 'Author ID'+'_x', 'Author ID'+'_y'])  # CHECKPOINT.
                    author_svnet = author_svnet.loc[author_svnet.loc[:, 'Author ID'+'_x'].values < author_svnet.loc[:, 'Author ID'+'_y'].values, :]  # Not allow loop edge and duplicate undirected edge.
                    author_svnet = author_svnet.groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')  # How many matched venues between author x and author y.
                if '6.1V2' in args.config:
                    # Consecutive operations to try to reduce internal intermediate memory use.
                    author_svnet = paper_align_i.loc[:, ['Unify venue', 'Year', 'Author ID']].drop_duplicates() \
                        .merge(paper_align_i.loc[:, ['Unify venue', 'Year', 'Author ID']].drop_duplicates(), left_on=['Unify venue', 'Year'], right_on=['Unify venue', 'Year']) \
                        .loc[:, ['Author ID'+'_x', 'Author ID'+'_y']]
                    if len(args.network) == 1:
                        paper_align_i = None  # Potentially clean up. When only run network 6.
                    author_svnet = author_svnet \
                        .loc[author_svnet.loc[:, 'Author ID'+'_x'].values < author_svnet.loc[:, 'Author ID'+'_y'].values, :] \
                        .groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')
                if '6.1V3' in args.config:
                    # Note: there is a simple way to reduce ram use: gradually process small parts of author list.
                    #   The actual work here is matching author-author through venue_year, so instead of matching all authors together,
                    #       we can split authors into small parts, then matching in each part and between parts.
                    #   If each part contains exclusive authors, author pairs will not be repeated after matching parts.
                    #       So we can finish and save result after each loop.
                    #   Note that this method is different to the SO for-loop solution:
                    #       it does not merge each authors but many authors at once using pandas, so it's faster but we still need to filter loop and repeated edges later.
                    # Moreover, we can filter author_spnet and weight in each loop,
                    #   we can also partially filter weight=1 sooner by discarding authors that go to only 1 venue_year ever.
                    # => Use much less ram and run much faster than fully pandas merging and any other configs.
                    # FREE LUNCH.
                    print('Try to reduce ram use on network 6!')
                    print('\t' + 'Only count each venue once for each author.')
                    paper_align_i = paper_align_i.loc[:, ['Unify venue', 'Year', 'Author ID']].drop_duplicates()  # in case authors have many papers at 1 venue_year, only consider once.
                    print('\t' + '\t' + 'Paper align size: ' + str(paper_align_i.shape[0]))
                    if 1 not in args.weight_thresholds:
                        print('\t' + 'Early partially filter out weight == 1.')
                        paper_align_i = paper_align_i.loc[paper_align_i.duplicated(subset='Author ID', keep=False), :]  # discard authors that appear only once.
                        print('\t' + '\t' + 'Paper align size: ' + str(paper_align_i.shape[0]))
                    num_parts = 10  # 55 loop.
                    print('\t' + 'Matching ' + str(num_parts) + ' small parts.')
                    sorted_authors = sorted(paper_align_i.loc[:, 'Author ID'].unique())  # Sort is not required, but it's still better, for later filtering.
                    part_size = np.ceil(len(sorted_authors) / num_parts).astype(int)
                    if min(args.weight_thresholds) > 1:
                        print('\t' + '\t' + 'Will filter out weight < ' + str(min(args.weight_thresholds)))
                    for part in range(num_parts):
                        for matching_part in range(part, num_parts):  # each part matches with itself and later parts.
                            print('\t' + '\t' + 'Matching part ' + str(part) + ' to part ' + str(matching_part))
                            left = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].isin(sorted_authors[part * part_size: (part + 1) * part_size]), :]
                            right = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].isin(sorted_authors[matching_part * part_size: (matching_part + 1) * part_size]), :]
                            author_svnet = left.merge(right, on=['Unify venue', 'Year']).loc[:, ['Author ID'+'_x', 'Author ID'+'_y']]
                            author_svnet = author_svnet \
                                .loc[author_svnet.loc[:, 'Author ID'+'_x'].values < author_svnet.loc[:, 'Author ID'+'_y'].values, :] \
                                .groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')
                            if min(args.weight_thresholds) > 1:
                                author_svnet = author_svnet.loc[author_svnet.loc[:, 'Size'] >= min(args.weight_thresholds), :]  # Size is final weight. Discard unused weights.
                            if 'NoneCoAuthor' in args.config:
                                # Do not build network between authors that have ever been coauthors even once, in any venue_year.
                                author_svnet = author_svnet.merge(author_spnet, on=['Author ID'+'_x', 'Author ID'+'_y'], how='left')
                                eval_cols(author_svnet, ['Author ID'+'_x', 'Author ID'+'_y', 'Size'+'_x', 'Size'+'_y'])  # CHECKPOINT.
                                author_svnet = author_svnet.loc[pd.isnull(author_svnet.loc[:, 'Size'+'_y'].values), ['Author ID'+'_x', 'Author ID'+'_y', 'Size'+'_x']]
                                for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                                    author_svnet.loc[author_svnet.loc[:, 'Size'+'_x'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'NoneCoAuthor', 'AUTHOR_SHARE_VENUE_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False, mode='a')
                            for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                                author_svnet.loc[author_svnet.loc[:, 'Size'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'AUTHOR_SHARE_VENUE_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False, mode='a')

            if any(x.startswith('6.2') for x in args.config):
                # 6.2 Option 2: Partial merge (SO for-loop solution).
                global sorted_authors_i
                sorted_authors_i = sorted(paper_align_i.loc[:, 'Author ID'].unique())  # series.unique(), return 1D numpy ndarray, faster numpy.unique() because using hashtable vs sort.
                # The algorithm is simple: Sort unique author id, then for each author id, only merge with the author ids larger than it.
                # Count in each loop to get the final result of each loop.
                if '6.2.1' in args.config:
                    # 6.2.1 Sequential partial merge: => too slow: million of slices and merges.
                    base_time = time.time()
                    result_list = []
                    for i in range(len(sorted_authors_i) - 1):
                        left = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].values == sorted_authors_i[i], ['Unify venue', 'Year', 'Author ID']].drop_duplicates()
                        right = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].values > sorted_authors_i[i], ['Unify venue', 'Year', 'Author ID']].drop_duplicates()
                        partial_result = left.merge(right, left_on=['Unify venue', 'Year'], right_on=['Unify venue', 'Year']) \
                            .groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')  # This merge also filters out loop edge and duplicate undirected edge. Because of how we iterate through a sorted unique author list, each authorx authory pair is not repeated, so the count in each loop is final result.
                        result_list.append(partial_result)
                    print('\t' + 'Sequential partial merge time: ' + str(time.time()-base_time)); base_time = time.time()
                # 6.2.2 Parallel partial merge: => out of memory: million of slices and merges in different processes, maybe it makes many data copies.
                if '6.2.2V1' in args.config:
                    base_time = time.time()
                    pool = multiprocessing.Pool(args.parallel_workers)
                    result_list = pool.map(func=partial_merge_a_sv, iterable=range(len(sorted_authors_i) - 1))  # chunksize default = len iterable / (4 * num process). Items in each chunk will be fed consecutively into a process, similarly to map in map-reduce. But it is still not working here.
                    pool.close()
                    pool.join()
                    print('\t' + 'multiprocessing.Pool partial merge time: ' + str(time.time()-base_time)); base_time = time.time()
                    # result_list = joblib.Parallel(n_jobs=args.parallel_workers)(joblib.delayed(function=partial_merge_a_sv)(i) for i in range(len(sorted_authors_i) - 1))
                    # print('\t' + 'joblib partial merge time: ' + str(time.time()-base_time)); base_time = time.time()
                if '6.2.2V2' in args.config:
                    # May need to try reducing the number of process and increase work in each process: pass a list of authors need to be merged to each process. -> each process loop through the list to build left, then concat and return only 1 df as an item in map result list. This will also balance the load between process.
                    passing_authors = []
                    i = 0
                    passing_size = 1
                    while True:
                        passing_authors.append(sorted_authors_i[i:i+passing_size])
                        i += passing_size
                        passing_size += 1
                        if i > len(sorted_authors_i):
                            break
                    base_time = time.time()
                    pool = multiprocessing.Pool(args.parallel_workers)
                    result_list = pool.map(func=partial_merge_a_sv_v2, iterable=passing_authors)  # chunksize default = len iterable / (4 * num process). Items in each chunk will be fed consecutively into a process, similarly to map in map-reduce. But it is still not working here.
                    print('\t' + 'multiprocessing.Pool partial merge time v2: ' + str(time.time()-base_time)); base_time = time.time()
                    pool.close()
                    pool.join()
                    # result_list = joblib.Parallel(n_jobs=args.parallel_workers)(joblib.delayed(function=partial_merge_a_sv_v2)(arg) for arg in passing_authors)
                    # print('\t' + 'joblib partial merge time v2: ' + str(time.time()-base_time)); base_time = time.time()
                # Concat and checkpoint for both v1 & v2.
                if len(result_list) > 0:
                    author_svnet = pd.concat(result_list)
                    base_time = time.time()
                    if not author_svnet.empty:
                        if author_svnet.duplicated(subset=['Author ID'+'_x', 'Author ID'+'_y']).values.any():  # CHECKPOINT.
                            raise ValueError('Share-venue author network is not properly counted.')
                    print('\t' + '\t' + 'Checkpoint pd.concat time: ' + str(time.time()-base_time)); base_time = time.time()
                else:
                    author_svnet = pd.DataFrame()

            # Common.
            if not author_svnet.empty:
                if '6.1V3' not in args.config:
                    # Config 6.1V3 already did all of these.
                    if 'NoneCoAuthor' in args.config:
                        # Do not build network between authors that have ever been coauthors even once, in any venue_year.
                        author_svnet = author_svnet.merge(author_spnet, on=['Author ID' + '_x', 'Author ID' + '_y'], how='left')
                        author_spnet = None  # Clean up right away.
                        eval_cols(author_svnet, ['Author ID' + '_x', 'Author ID' + '_y', 'Size' + '_x', 'Size' + '_y'])  # CHECKPOINT.
                        author_svnet = author_svnet.loc[ pd.isnull(author_svnet.loc[:, 'Size' + '_y'].values), ['Author ID' + '_x', 'Author ID' + '_y', 'Size' + '_x']]
                        # author_svnet = author_svnet.drop_duplicates()  # Just to make sure. But it will build a hastable with size of distinct items, without any duplicate the memory cost is too hight. Make sure not duplicate by code logic. Save ram here.
                        for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                            author_svnet.loc[author_svnet.loc[:, 'Size'+'_x'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'NoneCoAuthor', 'AUTHOR_SHARE_VENUE_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False)
                    else:
                        # author_svnet = author_svnet.drop_duplicates()  # Just to make sure. But it will build a hastable with size of distinct items, without any duplicate the memory cost is too hight. Make sure not duplicate by code logic. Save ram here.
                        for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                            author_svnet.loc[author_svnet.loc[:, 'Size'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'AUTHOR_SHARE_VENUE_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False)
                    print('\t' + '# Share-venue author net nodes: ' + str(len(pd.unique(author_svnet.loc[:, ['Author ID'+'_x', 'Author ID'+'_y']].values.ravel()))))  # CHECKPOINT.
                author_svnet = None
            print('\t' + 'Share-venue author net year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()

        if 7 in args.network:
            # 7. Share-author venue network.
            # Option 1: Fully merge.
            venue_sanet = paper_align_i.loc[:, ['Unify venue', 'Author ID']].drop_duplicates() \
                .merge(paper_align_i.loc[:, ['Unify venue', 'Author ID']].drop_duplicates(), left_on='Author ID', right_on='Author ID')  # Only count how many shared authors, not count how many times an author goes to the venue.
            eval_cols(venue_sanet, ['Unify venue'+'_x', 'Author ID', 'Unify venue'+'_y'])  # CHECKPOINT.
            venue_sanet = venue_sanet.loc[venue_sanet.loc[:, 'Unify venue'+'_x'].values < venue_sanet.loc[:, 'Unify venue'+'_y'].values, :]  # Not allow loop edge and duplicate undirected edge.
            venue_sanet = venue_sanet.groupby(['Unify venue'+'_x', 'Unify venue'+'_y'], sort=False).size().reset_index(name='Size')  # How many matched authors between venue x and venue y.
            # Common.
            if not venue_sanet.empty:
                eval_cols(venue_sanet, ['Unify venue'+'_x', 'Unify venue'+'_y', 'Size'])  # CHECKPOINT.
                # venue_sanet = venue_sanet.drop_duplicates()  # Just to make sure.
                for weight_threshold in args.weight_thresholds:  # e.g., at least weight 5.
                    venue_sanet.loc[venue_sanet.loc[:, 'Size'] >= weight_threshold, :].to_csv(os.path.join(root_path, args.save_dir, 'VENUE_SHARE_AUTHOR_NETWORK_' + str(test_year) + '_' + str(weight_threshold) + '.txt'), sep=' ', header=False, index=False)
                print('\t' + '# Share-author venue net nodes: ' + str(len(pd.unique(venue_sanet.loc[:, ['Unify venue'+'_x', 'Unify venue'+'_y']].values.ravel()))))  # CHECKPOINT.
                venue_sanet = None
            print('\t' + 'Share-author venue net year ' + str(test_year) + ': ' + str(time.time()-base_time)); base_time = time.time()

    print('SAVE NETWORKS: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))


def partial_merge_a_sv(i):
    global paper_align_i
    global sorted_authors_i

    left = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].values == sorted_authors_i[i], ['Unify venue', 'Year', 'Author ID']].drop_duplicates()
    right = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].values > sorted_authors_i[i], ['Unify venue', 'Year', 'Author ID']].drop_duplicates()
    partial_result = left.merge(right, left_on=['Unify venue', 'Year'], right_on=['Unify venue', 'Year']) \
        .groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')

    return partial_result


def partial_merge_a_sv_v2(authors):
    global paper_align_i

    result_list = []
    for author in authors:
        left = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].values == author, ['Unify venue', 'Year', 'Author ID']].drop_duplicates()
        right = paper_align_i.loc[paper_align_i.loc[:, 'Author ID'].values > author, ['Unify venue', 'Year', 'Author ID']].drop_duplicates()
        partial_result = left.merge(right, left_on=['Unify venue', 'Year'], right_on=['Unify venue', 'Year']) \
            .groupby(['Author ID'+'_x', 'Author ID'+'_y'], sort=False).size().reset_index(name='Size')
        result_list.append(partial_result)

    if len(result_list) > 0:
        big_partial_result = pd.concat(result_list)
    else:
        big_partial_result = pd.DataFrame()

    return big_partial_result


def eval_cols(df, cols):
    """Evaluate columns.
    Raise error if something is wrong. If not, return the columns.

    :param df:
    :param cols:
    :return: safe columns.
    """

    if len(cols) != len(df.columns.tolist()):
        print('DataFrame: ' + str(len(df.columns.tolist())) + '. Checked cols: ' + str(len(cols)))
        print('DataFrame columns: ' + str(df.columns.tolist()))
        print('Checked columns: ' + str(cols))
        raise KeyError('Wrong number of columns.')

    wrong_cols = []
    for col in cols:
        if col not in df.columns.tolist():
            wrong_cols.append(col)
    if len(wrong_cols) > 0:
        print('Wrong columns: ' + str(wrong_cols))
        print('DataFrame number of columns: ' + str(df.columns.tolist()))
        print('Checked number of columns: ' + str(cols))
        raise KeyError('Wrong columns labels.')

    return cols


def unify_venue(row):
    """Unify the venue ID.

    :param row: pandas row of paper.
    :return: 'J.' + JournalID or 'C.' + ConferenceID or None if no venue ID or both journal id and conf id.
    """

    if pd.notnull(row.loc['Journal ID']) and pd.notnull(row.loc['Conference ID']):
        return None
    elif pd.notnull(row.loc['Journal ID']):
        return 'J.' + str(row.loc['Journal ID'])
    elif pd.notnull(row.loc['Conference ID']):
        return 'C.' + str(row.loc['Conference ID'])
    else:
        return None


def divide_venue(row):
    """Divide the unified venue ID.

    :param row: pandas row of paper.
    :return: List [JournalID, ConferenceID].
    """

    if pd.isnull(row.loc['Unify_venue']):
        return [None, None]
    elif row.loc['Unify_venue'].startswith('J.'):
        return [row.loc['Unify_venue'][2:], None]
    elif row.loc['Unify_venue'].startswith('C.'):
        return [None, row.loc['Unify_venue'][2:]]
    else:
        raise ValueError('Unify venue format not recognized.')


def unify_venue_np(row, j, c):
    """Unify the venue ID.

    :param row: pandas row of paper.
    :param j: journal id position.
    :param c: conference id position.
    :return: 'J.' + JournalID or 'C.' + ConferenceID or None if no venue ID or both journal id and conf id.
    """

    if pd.notnull(row[j]) and pd.notnull(row[c]):
        return None
    elif pd.notnull(row[j]):
        return 'J.' + str(row[j])
    elif pd.notnull(row[c]):
        return 'C.' + str(row[c])
    else:
        return None


def divide_venue_np(row, v):
    """Divide the unified venue ID.

    :param row: pandas row of paper.
    :param v: unify venue id position.
    :return: List [JournalID, ConferenceID].
    """

    if pd.isnull(row[v]):
        return [None, None]
    elif row[v].startswith('J.'):
        return [row[v][2:], None]
    elif row[v].startswith('C.'):
        return [None, row[v][2:]]
    else:
        raise ValueError('Unify venue format not recognized.')


def read_x(paper_align_filename, paper_cit_filename=None, author_cit_filename=None, venue_cit_filename=None,
           paper_sa_filename=None, author_sp_filename=None, author_sv_filename=None, venue_sa_filename=None,
           pooling='avg', year=1996, trim_combine=True, fillna='avg'):
    """Read embedding input matrix.
    Align by paper id in paper align.

    Notice: all embeddings are filtered and aligned based on paper_align.
    -> Need to gurantee that paper has author and venue, then check alignment and fill zero to empty value.

    Note:
        - There are many embeddings. If it's not None, read it and combine to x.
        - After combining, separate to train and test. (remember to discard 5 years before test year, e.g., 1991-1995.)

    Modularize.

    :param paper_align_filename:
    :param paper_cit_filename:
    :param author_cit_filename:
    :param venue_cit_filename:
    :param paper_sa_filename:
    :param author_sp_filename:
    :param author_sv_filename:
    :param venue_sa_filename:
    :param pooling: {['avg'], 'sum', 'max'}.
    :param year: test year to divide train/test. Default 1996.
    :param trim_combine: [True] means trim papers when a combined embeddings lacking that paper id. False means keep all papers in paper_align, fillna in embedding by 0 or mean().
    :param fillna: how to fillna when not trim_combine: 'avg' or a value to fill in. Default: average.
    :return: matrixes train_x and test_x (numpy array).
    """

    print('READ X.')
    start_time = time.time()  # in second.

    # Read paper align.
    names_paper_align = ['Year', 'Paper ID', 'Author ID', 'Venue ID']
    dtype_paper_align = {'Year': int, 'Paper ID': str, 'Author ID': str, 'Venue ID': str}
    paper_align = pd.read_csv(paper_align_filename, delimiter=' ', header=None, skiprows=0, names=names_paper_align, dtype=dtype_paper_align, engine='c')
    paper_align = paper_align.loc[(paper_align.loc[:, 'Year'] < year - 5) | (paper_align.loc[:, 'Year'] == year), :]  # The order is maintained through later merge, drop_dupicate, iloc[] (but here we use merge instead of concat.)

    x = paper_align.loc[:, ['Paper ID']].drop_duplicates()

    # Read each embeddings and combine to x.
    # Modularizing: for each embedding file, read, compute, combine.
    # Data structure:
    #   first row: num samples num dim
    #   other rows: str paperid [embedding]
    if paper_cit_filename is not None:
        # Read.
        paper_cit = pd.read_csv(paper_cit_filename, delimiter=' ', header=None, skiprows=1, engine='c')
        paper_cit = paper_cit.rename(columns={paper_cit.columns[0]: 'Paper ID'})
        # Do not need to compute, because paper_cit has paper id already, just use.
        # Combine.
        if trim_combine:
            x = x.merge(paper_cit, on='Paper ID')
        else:
            x = x.merge(paper_cit, on='Paper ID', how='left')

    if author_cit_filename is not None:
        # Read.
        author_cit = pd.read_csv(author_cit_filename, delimiter=' ', header=None, skiprows=1, engine='c')
        author_cit = author_cit.rename(columns={author_cit.columns[0]: 'Author ID'})
        # Compute.
        author_cit = paper_align.loc[:, ['Paper ID', 'Author ID']].merge(author_cit, on='Author ID').drop('Author ID', axis=1)  # (no right join) Discard all embeddings with no matching info in paper_align, because do not know how to align them. (no left join) Discard all paper_align with no matching in embeddings, because this is just intermediate step, just keep what are in embeddings. So inner join.
        if pooling == 'avg':
            author_cit = author_cit.groupby('Paper ID').mean().reset_index()  # with mean() and sum(), column Author ID will disappear because of string type. reset_index() to put groupby key into the dataframe.
        elif pooling == 'sum':
            author_cit = author_cit.groupby('Paper ID').sum().reset_index()
        elif pooling == 'max':
            author_cit = author_cit.groupby('Paper ID').max().reset_index()
        # Combine.
        if trim_combine:
            x = x.merge(author_cit, on='Paper ID')
        else:
            x = x.merge(author_cit, on='Paper ID', how='left')

    if venue_cit_filename is not None:
        # Read.
        venue_cit = pd.read_csv(venue_cit_filename, delimiter=' ', header=None, skiprows=1, engine='c')
        venue_cit = venue_cit.rename(columns={venue_cit.columns[0]: 'Venue ID'})
        # Compute.
        venue_cit = paper_align.loc[:, ['Paper ID', 'Venue ID']].drop_duplicates().merge(venue_cit, on='Venue ID').drop('Venue ID', axis=1)
        # Combine.
        if trim_combine:
            x = x.merge(venue_cit, on='Paper ID')
        else:
            x = x.merge(venue_cit, on='Paper ID', how='left')

    if paper_sa_filename is not None:
        # Read.
        paper_sa = pd.read_csv(paper_sa_filename, delimiter=' ', header=None, skiprows=1, engine='c')
        paper_sa = paper_sa.rename(columns={paper_sa.columns[0]: 'Paper ID'})
        # Combine.
        if trim_combine:
            x = x.merge(paper_sa, on='Paper ID')
        else:
            x = x.merge(paper_sa, on='Paper ID', how='left')

    if author_sp_filename is not None:
        # Read.
        author_sp = pd.read_csv(author_sp_filename, delimiter=' ', header=None, skiprows=1, engine='c')
        author_sp = author_sp.rename(columns={author_sp.columns[0]: 'Author ID'})
        # Compute.
        author_sp = paper_align.loc[:, ['Paper ID', 'Author ID']].merge(author_sp, on='Author ID')
        if pooling == 'avg':
            author_sp = author_sp.groupby('Paper ID').mean().reset_index()
        elif pooling == 'sum':
            author_sp = author_sp.groupby('Paper ID').sum().reset_index()
        elif pooling == 'max':
            author_sp = author_sp.groupby('Paper ID').max().reset_index()
        # Combine.
        if trim_combine:
            x = x.merge(author_sp, on='Paper ID')
        else:
            x = x.merge(author_sp, on='Paper ID', how='left')

    if author_sv_filename is not None:
        # Read.
        author_sv = pd.read_csv(author_sv_filename, delimiter=' ', header=None, skiprows=1, engine='c')
        author_sv = author_sv.rename(columns={author_sv.columns[0]: 'Author ID'})
        # Compute.
        author_sv = paper_align.loc[:, ['Paper ID', 'Author ID']].merge(author_sv, on='Author ID')
        if pooling == 'avg':
            author_sv = author_sv.groupby('Paper ID').mean().reset_index()
        elif pooling == 'sum':
            author_sv = author_sv.groupby('Paper ID').sum().reset_index()
        elif pooling == 'max':
            author_sv = author_sv.groupby('Paper ID').max().reset_index()
        # Combine.
        if trim_combine:
            x = x.merge(author_sv, on='Paper ID')
        else:
            x = x.merge(author_sv, on='Paper ID', how='left')

    if venue_sa_filename is not None:
        # Read.
        venue_sa = pd.read_csv(venue_sa_filename, delimiter=' ', header=None, skiprows=1, engine='c')
        venue_sa = venue_sa.rename(columns={venue_sa.columns[0]: 'Venue ID'})
        # Compute.
        venue_sa = paper_align.loc[:, ['Paper ID', 'Venue ID']].drop_duplicates().merge(venue_sa, on='Venue ID').drop('Venue ID', axis=1)
        # Combine.
        if trim_combine:
            x = x.merge(venue_sa, on='Paper ID')
        else:
            x = x.merge(venue_sa, on='Paper ID', how='left')

    if fillna == 'avg':
        x = x.fillna(x.mean())
        x = x.fillna(0.0)  # fill 0 to columns containing all NA values.
    else:
        x = x.fillna(float(fillna))

    x = x.sort_values('Paper ID')  # Make result reproducible.

    print('READ X: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    return x.loc[x.loc[:, 'Paper ID'].isin(paper_align.loc[paper_align.loc[:, 'Year'] < year - 5, 'Paper ID'].values), :].values, \
        x.loc[x.loc[:, 'Paper ID'].isin(paper_align.loc[paper_align.loc[:, 'Year'] == year, 'Paper ID'].values), :].values


def read_y(x, num_year_step, citation_count_filename):
    """Read citation count output matrix for specific x matrix.
    Align by paper id in x matrix.
    Input x is numpy array type numpy.float64 (64 bit: 52 bits mantissa, 11 bits exponent).
    Output y is pandas dataframe based on numpy array type float (enough to store int 32 bit lossless, so ok.)
    Note: never convert int32 to numpy.float32: will lose precision.

    Options for updating y from citation_count:

    Option 1: Masking with iloc is too slow.
    y = pd.DataFrame(y)
    for row in citation_count:
        y.iloc[[y.iloc[:, 0] == row[0]], row[1]] = row[2]
    y = y.values

    Option 2: masking with numpy array: a little faster than pandas iloc.
    This is weird, seems pandas masking does not use index and overhead is high.
    for row in citation_count:
        y[[y[:, 0] == row[0]], row[1]] = row[2]

    Option 3: use 2 nested for-loop: way too slow.
    for row in citation_count:
        for j, rowj in enumerate(y):
            if rowj[0] == row[0]:
                y[j, row[1]] = row[2]

    Option 4: manually build index using dictionary: MUCH FASTER and guarantee to be correct.

    Option 5: pivoting: fast, similar to manually indexing. But the logic is obfuscated.
    citation_count_pivot = pd.DataFrame(citation_count)
    citation_count_pivot = citation_count_pivot.pivot_table(index=citation_count_pivot.columns[[0]].tolist(), columns=citation_count_pivot.columns[[1]].tolist(), values=citation_count_pivot.columns[[2]].tolist(), aggfunc=np.mean)  # Only make columns based on step years values, so if some step years are not present the columns will not be created.
    citation_count_pivot.columns = citation_count_pivot.columns.droplevel(0)
    citation_count_pivot = citation_count_pivot.reset_index(drop=False)
    y_pivot = pd.DataFrame(x[:, [0]])
    y_pivot = pd.merge(y_pivot, citation_count_pivot, left_on=y_pivot.columns[0], right_on=citation_count_pivot.columns[0], how='left')
    y_pivot = y_pivot.fillna(0.0)
    print('Check consistency pivot: ' + str((y_pivot.values == y).all().all()))

    :param x: (numpy matrix) containing ebeddings.
    :param num_year_step: length of citation count sequence.
    :param citation_count_filename:
    :return: y: (numpy array)
    """

    print('READ Y.')
    start_time = time.time()  # in second.

    num_samples = x.shape[0]

    # y: [numsamples][idpaper and 5 num_year citation count, padded 0]
    y = np.concatenate((x[:, [0]], np.zeros((num_samples, num_year_step))), axis=1)

    # Columns: paperid time citecount
    citation_count = pd.read_csv(citation_count_filename, delimiter=' ', header=None, skiprows=0).values

    # Option 4: manually create dictionary for idpaper and row number.
    # First, manually creating index using dictionary for idpaper to row number in y.
    paper2rownum = {}
    for i, rowi in enumerate(y):
        paper2rownum[rowi[0]] = i  # Paper ids in x and y are not repeated.
    # Then, update.
    for row in citation_count:
        if row[0] in paper2rownum:
            y[paper2rownum[row[0]], row[1]] = row[2]  # Step years in citation count file are not repeated.

    print('Check x y alignment: ' + str((x[:, 0] == y[:, 0]).all()))  # CHECKPOINT. Just to make sure.

    print('READ Y: DONE.')
    stop_time = time.time()
    print('Time (s): ' + str(stop_time-start_time))

    return y


def extract_merged_dataframe(df, num_col_df1, num_col_df2, col_df1=[], col_df2=[], joinkey_df1=[], joinkey_df2=[]):
    """Extract data in some columns from the merged dataframe. Check for different pandas merged result formats.

    :param df: merged dataframe.
    :param num_col_df1: number of columns in df1.
    :param num_col_df2: number of columns in df2.
    :param col_df1: (list of int) column position in df1 to keep (0-based).
    :param col_df2: (list of int) column position in df2 to keep (0-based).
    :param joinkey_df1:  (list of int) column position (0-based).
    :param joinkey_df2:  (list of int) column position (0-based).
    :return: extracted data from df.
    """

    return df.iloc[:, get_merged_column_index(num_col_df=df.shape[1], num_col_df1=num_col_df1, num_col_df2=num_col_df2, col_df1=col_df1, col_df2=col_df2, joinkey_df1=joinkey_df1, joinkey_df2=joinkey_df2)]


def get_merged_column_index(num_col_df, num_col_df1, num_col_df2, col_df1=[], col_df2=[], joinkey_df1=[], joinkey_df2=[]):
    """Transform the column indexes in old source dataframes to column indexes in merged dataframe. Check for different pandas merged result formats.

    :param num_col_df: number of columns in merged dataframe df.
    :param num_col_df1: number of columns in df1.
    :param num_col_df2: number of columns in df2.
    :param col_df1: (list of int) column position in df1 to keep (0-based).
    :param col_df2: (list of int) column position in df2 to keep (0-based).
    :param joinkey_df1:  (list of int) column position (0-based).
    :param joinkey_df2:  (list of int) column position (0-based).
    :return: (list of int) transformed column indexes, 0-based, in merged dataframe.
    """

    col_df1 = np.array(col_df1)
    col_df2 = np.array(col_df2)

    if num_col_df == num_col_df1 + num_col_df2:  # merging keeps same old columns
        col_df2 += num_col_df1
    elif num_col_df == num_col_df1 + num_col_df2 + 1:  # merging add column 'key_0' to the head
        col_df1 += 1
        col_df2 += num_col_df1 + 1
    elif num_col_df <= num_col_df1 + num_col_df2 - 1:  # merging deletes (possibly many) duplicated "join-key" columns in df2, keep and do not change order columns in df1.
        raise ValueError('Format of merged result is too complicated.')
    else:
        raise ValueError('Undefined format of merged result.')

    return np.concatenate((col_df1, col_df2)).astype(int).tolist()  # 1D numpy array is column vector, so concatenate by axis=0.


def parse_args():
    """Parses the arguments.
    """

    global debug  # claim to use global var.

    parser = argparse.ArgumentParser(description="Prepare data for citcount.")

    parser.add_argument('--root-path', default=None,
                        help="Root folder path. Default None.")

    parser.add_argument('--save-dir', default='CitCount',
                        help="Save folder. Default 'CitCount'.")

    parser.add_argument('--temp-dir', default='temp',
                        help='Temp folder. Default "temp".')

    parser.add_argument('--debug-server', dest='debug_server', action='store_true',
                        help='Turn on debug mode on server. Default: off.')
    parser.set_defaults(debug_server=False)

    parser.add_argument('--parallel-workers', type=int, default=multiprocessing.cpu_count() - 2,
                        help='Number of parallel jobs. Default cpu count - 2.')

    parser.add_argument('--start-test-year', type=int, default=1996,
                        help='The start test year. Default: 1996.')
    parser.add_argument('--end-test-year', type=int, default=2000,
                        help='The end test year. Default: 2000.')
    parser.add_argument('--min-year-step', type=int, default=1,
                        help='The min year step. Default: 1.')
    parser.add_argument('--max-year-step', type=int, default=5,
                        help='The max year step. Default: 5.')
    parser.add_argument('--min-year', type=int, default=1970,
                        help='The min year. Default: 1970.')

    parser.add_argument('--no-save-align', dest='save_align', action='store_false',
                        help='Do not save paper align. Default: save.')
    parser.set_defaults(save_align=True)

    parser.add_argument('--no-save-citcount', dest='save_citcount', action='store_false',
                        help='Do not save citation count sequence. Default: save.')
    parser.set_defaults(save_citcount=True)

    parser.add_argument('--ignore-citation', dest='ignore_citation', action='store_true',
                        help='Ignore citation in paper_paper when not save citcount and not compute citnetwork. Default: Not ignore.')
    parser.set_defaults(ignore_citation=False)

    parser.add_argument('--network', nargs='+', type=int, default=range(1, 8),
                        help='Networks to save: integer list, model from 1 to 7. Default: all.')

    parser.add_argument('--config', nargs='*', default=['6.1V3'],
                        help='Config choice to compute network 6.: string list, [6.1V1: simple fully merge, 6.1V2: consecutive fully merge, 6.1V3: try to reduce ram by gradually merge, 6.2.1: sequential partial merge, 6.2.2V1: parallel partial merge, 6.2.2V2: parallel partial merge auto balancing passing authors]. Default: 6.1V3.')

    parser.add_argument('--weight-thresholds', nargs='+', type=int, default=range(1, 6),
                        help='Threshold of weight to save: integer. Default: [1, 2, 3, 4, 5], will save 5 files for weight >= 1 to >= 5.')

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
        args.network = [6]

    # Finally return.
    return args


if __name__ == '__main__':
    debug = False
    paper_align_i = None
    sorted_authors_i = None
    main(parse_args())
