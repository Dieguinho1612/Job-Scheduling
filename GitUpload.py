#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import git


# In[ ]:


def upload_git_repo(DS):
    DS_str = "0"*(2-len(str(DS))) + str(DS)
    repo = git.Repo("")
    repo.git.add(f'Data/DataSet_{DS_str}')
    repo.git.commit(m=f'Upload Data Set {DS_str}')
    repo.git.push()
    shutil.rmtree(f'Data/DataSet_{DS_str}')

