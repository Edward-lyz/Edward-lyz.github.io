# 1. 背景
远程开发的服务器，经常会遇到以下的几个痛点：
- [ ] 远程启动了 docker 后，还得手动装一遍 icoding-agent，注册到 icoding 空间进行开发，多个开发机之间切换繁琐，多台机器间的代码状态不一致。
- [ ] 远程机器说没就没，哪怕你挂载了自己的 home 目录，照样说没就没，代码资料没有安全性的保证
- [ ] 回家开发时，内网 VPN 卡的不行，改个代码还得看网络状态
因此，我们提出这么一个想法：就把代码放在 mac 本机上，远程服务器只是提供一个运行代码的环境，且切换服务器开销也不大，把本地代码再传一次即可。为了做到这一步，我们提供两种思路：远程服务器开启 SFTP 服务，或者 WebDAV 服务，将文件系统暴露在 内网8000～9000 端口上（合规的），从而方便 RD 开发，且保证代码不丢。
# 2. 远端开启 SFTP 服务
为了简化大家的使用， 这里给一个脚本，能一键运行。
==注意：需要指定好自己的工作目录和密码！！！==
==注意：一定要在自己的容器内运行！！！==
- [ ] 第一步：下载脚本
```Shell
wget https://bj.bcebos.com/vllm-ai-models/liyanzhen01/start_proftpd.sh
```
- [ ] 第二步：执行脚本：
```Shell
sudo bash start_proftpd.sh --work-dir <your_path> --password <your_password>
```
顺利执行的话，会输出如下结果：
```Shell
ProFTPD 已启动
端口: 8xxx
配置: /etc/proftpd/proftpd.conf
日志: /var/log/proftpd/proftpd.log
```
配置文件的位置和日志位置均有提示，可以进一步查看～。到这里，服务器端就配置好了。

# 3.  Vscode 配置
远程配置好了后，我们就可以配置客户端的设置了，这里以 Vscode 为例，介绍如何配置。
- [ ] 第一步：下载插件：`natizyskunk.sftp` 
- [ ] 第二步：在当前的工作目录下，找到 `.vscode`的配置文件夹，如果没有，就新建一个
- [ ] 第三步，将如下的 json 配置，粘贴到`.vscode/sftp.json`文件中（新建一个）

```Json
{
    "protocol": "ftp",
    "port": 8xxx,
    "username": "root",
    "password": "<Your-Password>",
    "remotePath": "/",
    "uploadOnSave": true,
    "syncMode": "full",
    "watcher": {
        "files": "*",
        "autoUpload": true,
        "autoDelete": false
    },
    "ignore": [
        ".vscode",
        "node_modules",
        "models",
        ".DS_Store",
        ".devcontainer",
        ".venv"
    ],
    "passive": true,
    "debug": true,
    "retryOnError": true,
    "retryCount": 3,
    "retryDelay": 5000,
    "profiles": {
        "B200-dev": {
            "host": "tjzj-inf-sci-k8s-bzz2-0000.tjzj.baidu.com"
        },
        "B200-dev-2": {
            "host": "tjzj-inf-sci-k8s-bzz2-0183.tjzj.baidu.com"
        }
    },
    "defaultProfile": "B200-dev-2"
}
```
- [ ] 第四步：修改配置中的**端口，密码**，使得这里的配置能和第二小节的远程服务器的配置保持一致。
- [ ] 第五步：切换默认的远程服务器，vscode 这个插件一次可以展示一个远程服务器上的文件列表，如果想切换服务器用，可以切换 `defaultProfile` 这个字段对应的服务器名称。
# 4. 效果展示：

跟着上述步骤走一遍后，可以看到，在 vscode 的侧边栏有一个文件的图标，点开即可浏览远程的文件有哪些，可以右键选择下载到本地。如下图所示：
![[SFTP vscode 展示.png]]
同时，我们如果想将本地文件上传，可以右键你想上传的文件/文件夹，选择 `upload` 这项即可。如下图所示。
![[本地上传演示.png]]
同时，由于我们的配置，现在你本地浏览的任何文件，在你保存时，就会默认上传至远程服务器，对于本地开发，远端运行的模式来说，几乎等于无感。
至此，我们的配置到这里就结束了。