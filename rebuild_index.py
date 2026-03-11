#!/usr/bin/env python3
"""重建会话元数据缓存并压缩数据库

使用场景：
1. 删除旧版全文索引后的瘦身
2. 重新扫描会话文件并刷新元数据
3. 清理碎片空间
"""

import sqlite3
from pathlib import Path
from claude_viewer import DB_PATH, init_db, scan_session_files, index_file

def rebuild_database():
    """完全重建轻量元数据缓存"""
    print(f"数据库路径: {DB_PATH}")

    if DB_PATH.exists():
        backup_path = DB_PATH.with_suffix('.sqlite3.backup')
        print(f"备份原数据库到: {backup_path}")
        import shutil
        shutil.copy2(DB_PATH, backup_path)

    if DB_PATH.exists():
        print("删除旧数据库...")
        DB_PATH.unlink()

    print("创建新数据库...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    print("重新扫描会话文件...")
    total = 0
    for source, project, path, parent_session_id in scan_session_files():
        print(f"  扫描: {source}/{project}/{path.name}")
        index_file(conn, source, project, path, parent_session_id)
        total += 1

    conn.commit()

    print("压缩数据库...")
    conn.execute("VACUUM")

    conn.close()

    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n完成！")
    print(f"  扫描文件数: {total}")
    print(f"  数据库大小: {size_mb:.1f} MB")

if __name__ == "__main__":
    rebuild_database()
