# Script

You will find some scripts here. Some of them are useful, others aren't. Feel free to use them at your own discretion.

### Title Comment Generator

```bash
$ python title_comment_generator.py -c TEST
```

Gives you:

```bash
##################################### TEST #####################################
```

Beautiful, isn't it?

### Acces to Ipython notebook remotely

From remote server:

```bash
remote_user@remote_host$ ipython notebook --no-browser --port=8889
```

On local machine:

```bash
local_user@local_host$ ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
```

`-N`: no remote commands will be executed

`-f`: SSH go to background (we will need to kill the process)

`-L`: port forwarding configuration

Open your browser:

```
localhost:8888
```
[Source](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh)

