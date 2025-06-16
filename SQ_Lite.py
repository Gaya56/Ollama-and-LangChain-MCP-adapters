  def init_db():
      conn = sqlite3.connect('demo.db')
      cursor = conn.cursor()
      # Schema creation...
