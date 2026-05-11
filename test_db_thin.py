import oracledb
import os

# Caminhos da Wallet
wallet_dir = "/home/rdt/CameraIA/DriveOracle"
wallet_pem = os.path.join(wallet_dir, "ewallet.pem")

def test_db_thin():
    print(f"Iniciando teste em Modo Thin usando {wallet_pem}...")
    try:
        # String de conexão completa extraída do tnsnames.ora
        conn_str = '(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.sa-vinhedo-1.oraclecloud.com))(connect_data=(service_name=g674a77dea23c6a_imaculado_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))'
        
        conn = oracledb.connect(
            user="mensagem",
            password="crbsAcs@2026",
            dsn=conn_str,
            wallet_location=wallet_dir,
            wallet_password=None # Auto-login wallet
        )
        print("Conexão estabelecida com sucesso (Modo Thin)!")
        
        cursor = conn.cursor()
        
        # 1. Inserção de teste
        test_phone = "5511999999999"
        test_msg = "TESTE THIN MODE - CAMERA IA"
        cursor.execute("INSERT INTO DIZIMO.MENSAGENS (TELEFONE, TEXTO, STATUS, TIPO) VALUES (:1, :2, :3, :4)", 
                       [test_phone, test_msg, 0, 'G'])
        conn.commit()
        print("Registro de teste inserido com sucesso.")
        
        # 2. Consulta solicitada
        print("\n--- ULTIMOS REGISTROS EM DIZIMO.MENSAGENS ---")
        cursor.execute("SELECT TELEFONE, TEXTO, STATUS, DATA_CADASTRO FROM (SELECT * FROM DIZIMO.MENSAGENS ORDER BY DATA_CADASTRO DESC) WHERE ROWNUM <= 5")
        for row in cursor.fetchall():
            print(f"Fone: {row[0]} | Msg: {row[1]} | Status: {row[2]} | Data: {row[3]}")
            
        cursor.close()
        conn.close()
        print("\nTeste concluído.")
        
    except Exception as e:
        print(f"\nERRO NO MODO THIN: {e}")

if __name__ == "__main__":
    test_db_thin()
